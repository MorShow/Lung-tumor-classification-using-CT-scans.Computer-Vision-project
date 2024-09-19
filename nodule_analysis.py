import sys
import argparse
import datetime

import torch
import numpy as np
import scipy.ndimage.morphology as morphology
import scipy.ndimage.measurements as measurements

from utils import log
from utils.util import irc2xyz, xyz2irc
from datasets_classify import get_ct
from datasets_classify import candidate_info_tuple


class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            #log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
        )

        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='data/part2/models/seg_2020-01-26_19.45.12_w4d3c1-bal_1_nodupe-label_pos-d1_fn8-adam.best.state',
        )

        parser.add_argument('--cls-model',
            help="What to model class name to use for the classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default='data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
        )

        parser.add_argument('--malignancy-model',
            help="What to model class name to use for the malignancy classifier.",
            action='store',
            default='LunaModel',
            # default='ModifiedLunaModel',
        )
        parser.add_argument('--malignancy-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--tb-prefix',
            default='p2ch14',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("One and only one of series_uid and --run-validation should be given")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.init_model_path('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.init_model_path('cls')

        self.seg_model, self.cls_model, self.malignancy_model = self.init_models()

    def init_model_path(self, model_name):
        return 0
    def init_models(self):
        return 0, 0, 0

    def main(self):
        for _, series_uid in series_iter:
            ct = get_ct(series_uid)
            mask_a = self.segment_ct(series_uid, ct)

            candidate_info_list = self.group_segmentation_output(series_uid, ct, mask_a)
            classifications_list = self.classify_candidates(ct, candidate_info_list)

            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidate_info_dict:
                one_confusion = match_and_score(
                    classifications_list, candidate_info_dict[series_uid]
                )
                all_confusion += one_confusion
                print_confusion(
                    series_uid, one_confusion, self.malignancy_model is not None
                )

        print_confusion(
            "Total", all_confusion, self.malignancy_model is not None
        )

    def classify_candidates(self, ct, candidate_info_list):
        cls_dl = self.init_classification_dl(candidate_info_list)

        classifications_list = []

        for batch_idx, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_dev = input_t.to(self.device)
            with torch.no_grad():
                _, probability_nodule_dev = self.cls_model(input_dev)

                if self.malignancy_model is not None:
                    _, probability_mal_dev = self.malignancy_model(input_dev)
                else:
                    probability_mal_dev = torch.zeros_like(probability_nodule_dev)

            zip_iter = zip(center_list,
                           probability_nodule_dev[:, 1].tolist(),
                           probability_mal_dev[:, 1].tolist())

            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = irc2xyz(center_irc,
                                     direction_a=ct.direction_a,
                                     origin_xyz=ct.origin_xyz,
                                     vx_size_xyz=ct.vxSize_xyz
                                    )

                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)

        return classifications_list

    def segment_ct(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            seg_dl = self.init_segmentation_dl(series_uid)

            for input_t, _, _, slice_ndx_list in seg_dl:

                input_dev = input_t.to(self.device)
                prediction_dev = self.seg_model(input_dev)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_dev[i].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

        return mask_a

    def group_segmentation_output(self, series_uid, ct, clean_a):
        candidate_label_a, candidate_count = measurements.label(clean_a)

        center_irc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidate_label_a,
            index=np.arange(1, candidate_count + 1),
        )

        candidate_info_list = []

        for i, center_irc in enumerate(center_irc_list):
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )

            candidate_info_tup = candidate_info_tuple(False, False, False, 0.0, series_uid, center_xyz)
            candidate_info_list.append(candidate_info_tup)

        return candidate_info_list
