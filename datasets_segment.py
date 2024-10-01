import random
import functools
import os
import glob
import csv
from collections import namedtuple

import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset

from utils.util import xyz_tuple, xyz2irc
from utils.log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

candidate_info_tuple = namedtuple(
    'candidate_info_tuple',
    'is_nodule_bool, has_annotation_bool, is_mal_bool, diameter_mm, series_uid, center_xyz',
)

@functools.lru_cache(1)
def get_candidate_info_list(require_on_disk_bool=True):
    mhd_list = glob.glob('data/subset*/*.mhd')
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidate_info_list = []

    with open('data/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_disk_set:
                continue

            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            is_mal_bool = {'False': False, 'True': True}[row[5]]

            candidate_info_list.append(
                candidate_info_tuple(
                    True,
                    True,
                    is_mal_bool,
                    annotation_diameter_mm,
                    series_uid,
                    annotation_center_xyz,
                )
            )

    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in present_on_disk_set:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule_bool:
                candidate_info_list.append(
                    candidate_info_tuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidate_center_xyz,
                    )
                )

    candidate_info_list.sort(reverse=True)
    return candidate_info_list


@functools.lru_cache(1)
def get_candidate_info_dict(require_on_disk_bool=True):
    candidate_info_list = get_candidate_info_list(require_on_disk_bool)
    candidate_info_dict = {}

    for candidate_info_tup in candidate_info_list:
        candidate_info_dict.setdefault(candidate_info_tup.series_uid, []).append(candidate_info_tup)

    return candidate_info_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('data/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        self.hu_a.clip(-1000, 1000, self.hu_a)

        self.series_uid = series_uid

        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidate_info_list = get_candidate_info_dict()[self.series_uid]

        self.positive_info_list = [
            candidate_tup
            for candidate_tup in candidate_info_list
            if candidate_tup.is_nodule_bool
        ]

        self.positive_mask = self.build_annotation_mask(self.positive_info_list)
        self.positive_indices = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def build_annotation_mask(self, positive_info_list, threshold_hu=-700):
        bounding_box_a = np.zeros_like(self.hu_a, dtype=np.bool_)

        for candidateInfo_tup in positive_info_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            bounding_box_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1] = True

        mask_a = np.logical_and(bounding_box_a, (self.hu_a > threshold_hu))

        return mask_a

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        slice_list = []

        for axis, center_val in enumerate(center_irc):
            start_idx = int(round(center_val - width_irc[axis] / 2))
            end_idx = int(start_idx + width_irc[axis])

            if start_idx < 0:
                start_idx = 0
                end_idx = int(width_irc[axis])

            if end_idx > self.hu_a.shape[axis]:
                end_idx = self.hu_a.shape[axis]
                start_idx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_idx, end_idx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)


@functools.lru_cache(1, typed=True)
def get_ct_raw_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)

    ct_chunk.clip(-1000, 1000, ct_chunk)

    return ct_chunk, pos_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct_sample_size(series_uid):
    ct = Ct(series_uid)

    return int(ct.hu_a.shape[0]), ct.positive_indices


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_val_set_bool=None,
                 series_uid=None,
                 context_slices_count=3,
                 full_ct_bool=False
                 ):

        self.context_slices_count = context_slices_count
        self.full_ct_bool = full_ct_bool

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(get_candidate_info_dict().keys())

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indices = get_ct_sample_size(series_uid)

            if self.full_ct_bool:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indices]

        self.candidate_info_list = get_candidate_info_list()

        series_set = set(self.series_list)

        self.candidate_info_list = [cit for cit in self.candidate_info_list if cit.series_uid in series_set]
        self.pos_list = [nt for nt in self.candidate_info_list if nt.is_nodule_bool]

        log.info(f"{self!r}: {len(self.series_list)} {len(self.sample_list)} slices, {len(self.pos_list)} nodules")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        series_uid, slice_idx = self.sample_list[idx % len(self.sample_list)]
        return self.getitem_full_slice(series_uid, slice_idx)

    def getitem_full_slice(self, series_uid, slice_idx):
        ct = get_ct(series_uid)
        ct_t = torch.zeros((self.context_slices_count * 2 + 1, 512, 512))

        start_idx = slice_idx - self.context_slices_count
        end_idx = slice_idx + self.context_slices_count + 1

        for i, context_idx in enumerate(range(start_idx, end_idx)):
            context_idx = max(context_idx, 0)
            context_idx = min(context_idx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_idx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_idx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_idx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 60000

    def shuffle_samples(self):
        random.shuffle(self.candidate_info_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, idx):
        candidate_info_tup = self.pos_list[idx % len(self.pos_list)]
        return self.getitem_training_crop(candidate_info_tup)

    def getitem_training_crop(self, candidate_info_tup):
        ct_a, pos_a, center_irc = get_ct_raw_candidate(candidate_info_tup.series_uid,
                                                       candidate_info_tup.center_xyz,
                                                       (7, 96, 96))

        pos_a = pos_a[3:4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)

        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset + 64,
                                col_offset:col_offset + 64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset + 64,
                                 col_offset:col_offset + 64]).to(torch.long)

        slice_idx = center_irc.index

        return ct_t, pos_t, candidate_info_tup.series_uid, slice_idx
