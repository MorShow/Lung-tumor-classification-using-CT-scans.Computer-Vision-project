import argparse
import glob
import hashlib
import math
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measure
import scipy.ndimage.morphology as morph

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from utils.util import enumerate_with_estimate
from datasets_segment import (Luna2dSegmentationDataset, get_ct, get_candidate_info_list,
                              get_candidate_info_dict, candidate_info_tuple)
from datasets_classify import LunaDataset
from model_segment import UnetWrapper
from model_classify import LunaModel

from utils.log import logging
from utils.util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class FalsePosRateCheckApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int
        )

        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int
        )

        parser.add_argument('--series-uid',
            help='Limit inference to this Series UID only.',
            default=None,
            type=str
        )

        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default=None
        )

        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default=None
        )

        parser.add_argument('--tb-prefix',
            default='p2ch13',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        self.cli_args = parser.parse_args(sys_argv)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.init_model_path('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.init_model_path('cls')

        self.seg_model, self.cls_model = self.init_models()

    def init_model_path(self, type_str):
        pretrained_path = os.path.join(
            'models',
            type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
        )

        file_list = glob.glob(pretrained_path)
        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([pretrained_path, file_list])
            raise

    def init_models(self):
        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.debug(self.cli_args.segmentation_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UnetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        with open(self.cli_args.classification_path, 'rb') as f:
            log.debug(self.cli_args.classification_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        cls_dict = torch.load(self.cli_args.classification_path)

        cls_model = LunaModel()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model = seg_model.to(self.device)
            cls_model = cls_model.to(self.device)

        self.conv_list = nn.ModuleList([
            self._make_circle_conv(radius).to(self.device) for radius in range(1, 8)
        ])

        return seg_model, cls_model

    def init_segmentation_dl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
            context_slices_count=3,
            series_uid=series_uid,
            full_ct_bool=True,
        )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def init_classification_dl(self, candidate_info_list):
        cls_ds = LunaDataset(
            sortby_str='series_uid',
            candidate_info_list=candidate_info_list,
        )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        val_ds = LunaDataset(
            val_stride=10,
            is_val_set_bool=True,
        )
        val_set = set(
            candidate_info_tup.series_uid
            for candidate_info_tup in val_ds.candidate_info_list
        )

        positive_set = set(
            candidate_info_tup.series_uid
            for candidate_info_tup in get_candidate_info_list()
            if candidate_info_tup.is_nodule_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidate_info_tup.series_uid
                for candidate_info_tup in get_candidate_info_list()
            )

        train_list = sorted(series_set - val_set) if self.cli_args.include_train else []
        val_list = sorted(series_set & val_set)

        total_tp = total_tn = total_fp = total_fn = 0
        total_missed_pos = 0
        missed_pos_dist_list = []
        missed_pos_cit_list = []

        candidate_info_dict = get_candidate_info_dict()

        series_iter = enumerate(val_list + train_list)

        for _series_idx, series_uid in series_iter:
            ct, _output_dev, _mask_dev, clean_dev = self.segment_ct(series_uid)

            seg_candidate_info_list, _seg_center_irc_list, _ = self.cluster_segmentation_output(
                series_uid,
                ct,
                clean_dev,
            )

            if not seg_candidate_info_list:
                continue

            cls_dl = self.init_classification_dl(seg_candidate_info_list)
            results_list = []

            for batch_idx, batch_tup in enumerate(cls_dl):
                input_t, label_t, index_t, series_list, center_t = batch_tup

                input_dev = input_t.to(self.device)

                with torch.no_grad():
                    _logits_dev, probability_dev = self.cls_model(input_dev)

                probability_t = probability_dev.to('cpu')

                for i, _series_uid in enumerate(series_list):
                    assert series_uid == _series_uid, repr(
                        [batch_idx, i, series_uid, _series_uid, seg_candidate_info_list])

                    results_list.append((center_t[i], probability_t[i, 0].item()))

            # This part is all about matching up annotations with our segmentation results
            tp = tn = fp = fn = 0
            missed_pos = 0

            ct = get_ct(series_uid)

            candidate_info_list = candidate_info_dict[series_uid]
            candidate_info_list = [cit for cit in candidate_info_list if cit.is_nodule_bool]

            found_cit_list = [None] * len(results_list)

            for candidate_info_tup in candidate_info_list:
                min_dist = (999, None)

                for result_idx, (result_center_irc_t, nodule_probability_t) in enumerate(results_list):
                    result_center_xyz = irc2xyz(result_center_irc_t, ct.origin_xyz, ct.vxSize_xyz, ct.direction_a)
                    delta_xyz_t = torch.tensor(result_center_xyz) - torch.tensor(candidate_info_tup.center_xyz)
                    distance_t = (delta_xyz_t ** 2).sum().sqrt()

                    min_dist = min(min_dist, (distance_t, result_idx))

                distance_cutoff = max(10, candidate_info_tup.diameter_mm / 2)

                if min_dist[0] < distance_cutoff:
                    found_dist, result_idx = min_dist
                    nodule_probability_t = results_list[result_idx][1]

                    assert candidate_info_tup.is_nodule_bool

                    if nodule_probability_t > 0.5:
                        tp += 1
                    else:
                        fn += 1

                    found_cit_list[result_idx] = candidate_info_tup
                else:
                    log.warning("!!! Missed positive {}; {} min dist !!!".format(candidate_info_tup, min_dist))
                    missed_pos += 1
                    missed_pos_dist_list.append(float(min_dist[0]))
                    missed_pos_cit_list.append(candidate_info_tup)

            log.info("{}: {} missed pos, {} fn, {} fp, {} tp, {} tn".format(series_uid, missed_pos, fn, fp, tp, tn))
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_missed_pos += missed_pos

        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.info(self.cli_args.segmentation_path)
            log.info(hashlib.sha1(f.read()).hexdigest())
        with open(self.cli_args.classification_path, 'rb') as f:
            log.info(self.cli_args.classification_path)
            log.info(hashlib.sha1(f.read()).hexdigest())

        log.info("{}: {} missed pos, {} fn, {} fp, {} tp, {} tn".format('total', total_missed_pos, total_fn, total_fp,
                                                                        total_tp, total_tn))
        # missed_pos_dist_list.sort()
        # log.info("missed_pos_dist_list {}".format(missed_pos_dist_list))
        for cit, dist in zip(missed_pos_cit_list, missed_pos_dist_list):
            log.info("Missed by {}: {}".format(dist, cit))

    def segment_ct(self, series_uid):
        with torch.no_grad():
            ct = get_ct(series_uid)

            output_dev = torch.zeros(ct.hu_a.shape, dtype=torch.float32, device=self.device)

            seg_dl = self.init_segmentation_dl(series_uid)

            for batch_tup in seg_dl:
                input_t, label_t, series_list, slice_idx_list = batch_tup

                input_dev = input_t.to(self.device)
                prediction_dev = self.seg_model(input_dev)

                for i, slice_idx in enumerate(slice_idx_list):
                    output_dev[slice_idx] = prediction_dev[i, 0]

            mask_dev = output_dev > 0.5
            clean_dev = self.erode(mask_dev.unsqueeze(0).unsqueeze(0), 1)[0][0]

        return ct, output_dev, mask_dev, clean_dev

    def _make_circle_conv(self, radius):
        diameter = 1 + radius * 2

        a = torch.linspace(-1, 1, steps=diameter) ** 2
        b = (a[None] + a[:, None]) ** 0.5

        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv3d(1, 1, kernel_size=(1, diameter, diameter), padding=(0, radius, radius), bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv

    def erode(self, input_mask, radius, threshold=1):
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        return result >= threshold

    def cluster_segmentation_output(self, series_uid, ct, clean_dev):
        clean_a = clean_dev.cpu().numpy()

        candidate_label_a, candidate_count = measure.label(clean_a)
        center_irc_list = measure.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidate_label_a,
            index=list(range(1, candidate_count + 1)),
        )

        candidate_info_list = []

        for i, center_irc in enumerate(center_irc_list):
            assert np.isfinite(center_irc).all(), repr(
                [series_uid, i, candidate_count, (ct.hu_a[candidate_label_a == i + 1]).sum(), center_irc])
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            diameter_mm = 0.0

            candidate_info_tup = \
                candidate_info_tuple(None, None, None, diameter_mm, series_uid, center_xyz)
            candidate_info_list.append(candidate_info_tup)

        return candidate_info_list, center_irc_list, candidate_label_a
