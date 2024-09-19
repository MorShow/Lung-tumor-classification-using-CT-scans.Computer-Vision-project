import random
import math
import functools
import os
import glob
import csv
import copy
from collections import namedtuple

import torch
import torch.nn.functional as F
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
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vx_size_xyz, self.direction_a)

        slice_list = []

        for axis, center_val in enumerate(center_irc):
            start_idx = int(round(center_val - width_irc[axis]/2))
            end_idx = int(start_idx + width_irc[axis])

            if start_idx < 0:
                start_idx = 0
                end_idx = int(width_irc[axis])

            if end_idx > self.hu_a.shape[axis]:
                end_idx = self.hu_a.shape[axis]
                start_idx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_idx, end_idx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_ct(series_uid):
    return Ct(series_uid)


@functools.lru_cache(1, typed=True)
def get_ct_raw_candidate(series_uid, center_xyz, width_irc):
    ct = get_ct(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)

    return ct_chunk, center_irc


def get_ct_augmented_candidate(augmentation_dict, series_uid, center_xyz, width_irc, use_cache=True):
    if use_cache:
        ct_chunk, center_irc = get_ct_raw_candidate(series_uid, center_xyz, width_irc)
    else:
        ct = get_ct(series_uid)
        ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = random.random() * 2 - 1
            transform_t[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(transform_t[:3].unsqueeze(0).to(torch.float32),
                             list(ct_t.size()),
                             align_corners=False
                             )

    augmented_chunk = F.grid_sample(ct_t,
                                    affine_t,
                                    padding_mode='border',
                                    align_corners=False
                                    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 is_val_set_bool=False,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidate_info_list=None
                 ):

        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidate_info_list:
            self.candidate_info_list = copy.copy(candidate_info_list)
            self.use_cache = False
        else:
            self.candidate_info_list = copy.copy(get_candidate_info_list())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidate_info_tup.series_uid
                                          for candidate_info_tup in self.candidate_info_list))

        if is_val_set_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.candidate_info_list = [x for x in self.candidate_info_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidate_info_list)
        elif sortby_str == 'series_uid':
            self.candidate_info_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = [nt for nt in self.candidate_info_list if not nt.is_nodule_bool]
        self.pos_list = [nt for nt in self.candidate_info_list if nt.is_nodule_bool]
        self.ben_list = [nt for nt in self.pos_list if not nt.is_mal_bool]
        self.mal_list = [nt for nt in self.pos_list if nt.is_mal_bool]

        log.info(f"{self!r}: {len(self.candidate_info_list)} {'validation' if is_val_set_bool else 'training'} samples"
                 + f"{len(self.neg_list)} neg, {len(self.pos_list)} pos,"
                 + f"{self.ratio_int if self.ratio_int else 'unbalanced'} ratio")

    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.candidate_info_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            # We’re no longer tied to a specific number of samples, and presenting “a full epoch”
            # does not really make sense when we would have to repeat positive samples many, many
            # times to present a balanced training set. By picking 40,000 samples, we reduce the
            # time between starting a training run and seeing results

            return 10000
        else:
            return len(self.candidate_info_list)

    def __getitem__(self, idx):
        if self.ratio_int:
            pos_idx = idx // (self.ratio_int + 1)

            if idx % (self.ratio_int + 1):
                neg_idx = idx - 1 - pos_idx

                neg_idx = neg_idx % len(self.neg_list)
                candidate_info_tup = self.neg_list[neg_idx]
            else:
                pos_idx = pos_idx % len(self.pos_list)
                candidate_info_tup = self.pos_list[pos_idx]
        else:
            candidate_info_tup = self.candidate_info_list[idx]

        return self.sample_from_candidate_info_tup(candidate_info_tup, candidate_info_tup.is_nodule_bool)

    def sample_from_candidate_info_tup(self, candidate_info_tup, label_bool):
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = get_ct_augmented_candidate(self.augmentation_dict,
                                                                 candidate_info_tup.series_uid,
                                                                 candidate_info_tup.center_xyz,
                                                                 width_irc,
                                                                 self.use_cache
                                                                 )
        elif self.use_cache:
            candidate_a, center_irc = get_ct_raw_candidate(candidate_info_tup.series_uid,
                                                           candidate_info_tup.center_xyz,
                                                           width_irc
                                                           )

            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(candidate_info_tup.series_uid)

            candidate_a, center_irc = ct.get_raw_candidate(candidate_info_tup.center_xyz,
                                                           width_irc
                                                           )

            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidate_info_tup.series_uid, torch.tensor(center_irc)


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 20000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, idx):
        if self.ratio_int:
            if idx % 2 != 0:
                candidate_info_tup = self.mal_list[(idx // 2) % len(self.mal_list)]
            elif idx % 4 == 0:
                candidate_info_tup = self.ben_list[(idx // 4) % len(self.ben_list)]
            else:
                candidate_info_tup = self.neg_list[(idx // 4) % len(self.neg_list)]
        else:
            if idx >= len(self.ben_list):
                candidate_info_tup = self.mal_list[idx - len(self.ben_list)]
            else:
                candidate_info_tup = self.ben_list[idx]

        return self.sample_from_candidate_info_tup(candidate_info_tup, candidate_info_tup.is_mal_bool)
