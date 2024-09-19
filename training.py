import sys
import os
import shutil
import hashlib
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.log import logging
from utils.util import enumerate_with_estimate
from datasets_classify import get_ct
from datasets_segment import TrainingLuna2dSegmentationDataset, Luna2dSegmentationDataset
from model_segment import UnetWrapper, SegmentationAugmentation
from model_classify import LunaModel
import model_classify
import datasets_classify

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_IDX = 0
METRICS_PRED_IDX = 1
METRICS_PRED_P_IDX = 2
METRICS_LOSS_IDX = 3

METRICS_TP_IDX = 4
METRICS_FN_IDX = 5
METRICS_FP_IDX = 6

METRICS_SIZE = 7


class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
                            default=24,
                            help="Batch size to use for training.",
                            type=int
                            )

        parser.add_argument('--num-workers',
                            default=8,
                            help="Number of worker processes for background data loading.",
                            type=int
                            )

        parser.add_argument('--epochs',
                            default=1,
                            help="Number of epochs to train for.",
                            type=int
                            )

        parser.add_argument('--dataset',
                            help="What to dataset to feed the model.",
                            action='store',
                            default='LunaDataset',
                            )

        parser.add_argument('--model',
                            help="What to model class name to use.",
                            action='store',
                            default='LunaModel',
                            )

        parser.add_argument('--malignant',
                            help="Train the model to classify nodules as benign or malignant.",
                            action='store_true',
                            default=False,
                            )

        parser.add_argument('--finetune',
                            help="Start finetuning from this model.",
                            default='',
                            )

        parser.add_argument('--finetune-depth',
                            help="Number of blocks (counted from the head) to include in finetuning",
                            type=int,
                            default=1,
                            )

        parser.add_argument('--tb-prefix',
                            default='',
                            help="Data prefix to use for Tensorboard run.",
                            type=str
                            )

        parser.add_argument('--comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            type=str
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.train_writer = None
        self.val_writer = None
        self.total_training_samples_count = 0

        self.augmentation_dict = {}
        if True:
            # if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
            # if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
            # if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
            # if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
            # if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        model_cls = getattr(model_classify, self.cli_args.model)
        model = model_cls()

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')

            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]

            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"fine-tuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")

            model.load_state_dict({k: v for k, v in d['model_state'].items()
                                   if k.split('.')[0] not in model_blocks[-1]},
                                  strict=False
                                  )

            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)

            if self.use_cuda:
                log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))

                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model)

                model = model.to(self.device)

            return model

    def init_optimizer(self):
        lr = 0.003 if self.cli_args.finetune else 0.001

        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def init_train_dl(self):
        ds_cls = getattr(datasets_classify, self.cli_args.dataset)

        train_ds = ds_cls(
            val_stride=10,
            is_val_set_bool=False,
            ratio_int=1,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def init_val_dl(self):
        ds_cls = getattr(datasets_classify, self.cli_args.dataset)

        val_ds = ds_cls(
            val_stride=10,
            is_val_set_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def init_tensorboard_writers(self):
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(log_dir=log_dir + '-train_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        best_score = 0.0
        self.validation_cadence = 5 if not self.cli_args.finetune else 1

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        for epoch_idx in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch_idx} of {self.cli_args.epochs},"
                     + f"{len(train_dl)}/{len(val_dl)} batches of size {self.cli_args.batch_size}"
                     + f"*{(torch.cuda.device_count() if self.use_cuda else 1)}")

            train_metrics = self.do_training(epoch_idx, train_dl)
            self.log_metrics(epoch_idx, 'train', train_metrics)

            if epoch_idx == 1 or epoch_idx % self.validation_cadence == 0:
                val_metrics = self.do_validation(epoch_idx, val_dl)
                score = self.log_metrics(epoch_idx, 'val', val_metrics)
                best_score = max(score, best_score)

                self.save_model('cls', epoch_idx, score == best_score)

        if hasattr(self, 'train_writer'):
            self.train_writer.close()
            self.val_writer.close()

    def do_training(self, epoch_idx, train_dl):
        self.model.train()
        train_dl.dataset.shuffle_samples()

        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        ''' 
        batch_iter = enumerate_with_estimate(
            train_dl,
            f"Epoch {epoch_idx} training",
            start_idx=train_dl.num_workers,
        )
        '''

        batch_iter = enumerate(train_dl)

        for batch_idx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_idx,
                batch_tup,
                train_dl.batch_size,
                train_metrics,
                augment=True
            )

            loss_var.backward()
            self.optimizer.step()

        self.total_training_samples_count += len(train_dl.dataset)

        return train_metrics.to('cpu')

    def do_validation(self, epoch_idx, val_dl):
        with torch.no_grad():
            self.model.eval()

            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            '''
            batch_iter = enumerate_with_estimate(
                val_dl,
                f"Epoch {epoch_idx} training",
                start_idx=val_dl.num_workers,
            )
            '''

            batch_iter = enumerate(val_dl)

            for batch_idx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_idx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics,
                    augment=True
                )

            return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_idx, batch_tup, batch_size, metrics, augment=True):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_dev = input_t.to(self.device, non_blocking=True)
        label_dev = label_t.to(self.device, non_blocking=True)
        index_dev = index_t.to(self.device, non_blocking=True)

        if augment:
            input_dev = model_classify.augment3d(input_dev)

        logits_dev, probability_dev = self.model(input_dev)

        loss_dev = nn.functional.cross_entropy(logits_dev, label_dev[:, 1], reduction="none")

        start_idx = batch_idx * batch_size
        end_idx = start_idx + label_t.size(0)

        _, pred_label_dev = torch.max(probability_dev, dim=1, keepdim=False, out=None)

        metrics[METRICS_LABEL_IDX, start_idx:end_idx] = index_dev
        metrics[METRICS_PRED_IDX, start_idx:end_idx] = pred_label_dev
        metrics[METRICS_PRED_P_IDX, start_idx:end_idx] = probability_dev[:, 1]
        metrics[METRICS_LOSS_IDX, start_idx:end_idx] = loss_dev

        return loss_dev.mean()

    def log_metrics(self, epoch_idx, mode_str, metrics, classification_threshold=0.5):
        bins = np.linspace(0, 1)

        self.init_tensorboard_writers()

        log.info("E{} {}".format(
            epoch_idx,
            type(self).__name__,
        ))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'

        neg_label_mask = metrics[METRICS_LABEL_IDX] == 0
        neg_pred_mask = metrics[METRICS_PRED_IDX] == 0
        pos_label_mask = metrics[METRICS_LABEL_IDX] == 1
        pos_pred_mask = metrics[METRICS_PRED_IDX] == 1

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())
        neg_correct = int((np.logical_and(neg_label_mask, neg_pred_mask)).sum())
        pos_correct = int((np.logical_and(pos_label_mask, pos_pred_mask)).sum())

        tp_count = pos_correct
        tn_count = neg_correct
        fp_count = neg_count - neg_correct
        fn_count = pos_count - pos_correct

        metrics_dict = dict()

        metrics_dict['loss/all'] = metrics[METRICS_LOSS_IDX].mean()
        metrics_dict['loss/neg'] = metrics[METRICS_LOSS_IDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics[METRICS_LOSS_IDX, pos_label_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics.shape[1] * 100
        metrics_dict['correct/neg'] = neg_correct / neg_count * 100
        metrics_dict['correct/pos'] = pos_correct / pos_count * 100

        precision = metrics_dict['pr/precision'] = tp_count / np.float64(tp_count + fp_count)
        recall = metrics_dict['pr/recall'] = tp_count / np.float64(tp_count + fn_count)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0)

        tpr = (metrics[None, METRICS_PRED_P_IDX, pos_label_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics[None, METRICS_PRED_P_IDX, neg_label_mask] >= threshold[:, None]).sum(1).float() / neg_count

        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2

        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            writer.add_scalar(key, value, self.total_training_samples_count)

        fig = plt.figure()

        plt.plot(fpr, tpr)

        writer.add_figure('roc', fig, self.total_training_samples_count)
        writer.add_scalar('auc', auc, self.total_training_samples_count)

        log.info(
            (f"E{epoch_idx} {mode_str:10} {metrics_dict['loss/all']:.4f} loss, "
             + f"{metrics_dict['correct/all']:-5.1f}% correct, "
             + f"{metrics_dict['pr/precision']:.4f} precision, "
             + f"{metrics_dict['pr/recall']:.4f} recall, "
             + f"{metrics_dict['pr/f1_score']:.4f} f1 score, "
             + f"{metrics_dict['auc']:.4f} auc"
             )
        )
        log.info(
            (f"E{epoch_idx} {mode_str + '_' + neg:10} {metrics_dict['loss/neg']:.4f} loss, "
             + f"{metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct:} of {neg_count:})")
        )
        log.info(
            (f"E{epoch_idx} {mode_str + '_' + pos:10} {metrics_dict['loss/pos']:.4f} loss, "
             + f"{metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct:} of {pos_count:})")
        )

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics[METRICS_PRED_P_IDX, neg_label_mask],
            self.total_training_samples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics[METRICS_PRED_P_IDX, pos_label_mask],
            self.total_training_samples_count,
            bins=bins
        )

    def save_model(self, type_str, epoch_idx, is_best=False):
        file_path = os.path.join(
            'models',
            f'{type_str}_{self.time_str}_{self.cli_args.comment}.{self.total_training_samples_count}.state'
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_idx,
            'totalTrainingSamples_count': self.total_training_samples_count,
        }
        torch.save(state, file_path)

        log.debug(f"Saved model params to {file_path}")

        if is_best:
            best_path = os.path.join(
                'models',
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.cli_args.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug(f"Saved model params to {best_path}")

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
                            default=16,
                            help="Batch size to use for training.",
                            type=int
                            )

        parser.add_argument('--num-workers',
                            default=8,
                            help="Number of worker processes for background data loading.",
                            type=int
                            )

        parser.add_argument('--epochs',
                            default=1,
                            help="Number of epochs to train for.",
                            type=int
                            )

        parser.add_argument('--tb-prefix',
                            default='',
                            help="Data prefix to use for Tensorboard run.",
                            type=str
                            )

        parser.add_argument('--comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            type=str
                            )

        parser.add_argument('--balanced',
                            help="Balance the training data to half positive, half negative.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--flip',
                            help="Augment the training data by randomly flipping the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--offset',
                            help="Augment the training data by randomly offsetting the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--scale',
                            help="Augment the training data by randomly scaling the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--rotate',
                            help="Augment the training data by randomly rotating the data around X-Y axis.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--noise',
                            help="Augment the training data by adding the random noise to the data.",
                            action='store_true',
                            default=False
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.augmentation_dict = {}

        if self.cli_args.augmented or self.cli_args.flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.noise:
            self.augmentation_dict['noise'] = 25.0

        self.train_writer = None
        self.val_writer = None
        self.total_training_samples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_model(self):
        segmentation_model = UnetWrapper(in_channels=7,
                                         n_classes=1,
                                         depth=3,
                                         wf=4,
                                         padding=True,
                                         batch_norm=True,
                                         up_mode='upconv',
                                         )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def init_optimizer(self):
        return optim.Adam(self.segmentation_model.parameters())

    def init_train_dl(self):
        train_ds = TrainingLuna2dSegmentationDataset(val_stride=10,
                                                     is_val_set_bool=False,
                                                     context_slices_count=3)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda
                              )

        return train_dl

    def init_val_dl(self):
        val_ds = Luna2dSegmentationDataset(val_stride=10,
                                           is_val_set_bool=True,
                                           context_slices_count=3)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda
                            )

        return val_dl

    def init_tensorboard_writers(self):
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(log_dir=log_dir + '-train_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        best_score = 0.0
        self.validation_cadence = 5

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        for epoch_idx in range(1, self.cli_args.epochs + 1):
            train_metrics = self.do_training(epoch_idx, train_dl)
            self.log_metrics(epoch_idx, 'train', train_metrics)

            if epoch_idx == 1 or epoch_idx % self.validation_cadence == 0:
                val_metrics = self.do_validation(epoch_idx, val_dl)
                score = self.log_metrics(epoch_idx, 'val', val_metrics)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_idx, score == best_score)

                self.log_images(epoch_idx, 'train', train_dl)
                self.log_images(epoch_idx, 'val', val_dl)

        self.train_writer.close()
        self.val_writer.close()

    def do_training(self, epoch_idx, train_dl):
        self.segmentation_model.train()

        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        ''' 
        batch_iter = enumerate_with_estimate(
            train_dl,
            f"Epoch {epoch_idx} training",
            start_idx=train_dl.num_workers,
        )
        '''

        batch_iter = enumerate(train_dl)

        for batch_idx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_idx,
                batch_tup,
                train_dl.batch_size,
                train_metrics
            )

            loss_var.backward()
            self.optimizer.step()

        self.total_training_samples_count += train_metrics.size(1)

        return train_metrics.to('cpu')

    def do_validation(self, epoch_idx, val_dl):
        with torch.no_grad():
            self.segmentation_model.eval()

            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            '''
            batch_iter = enumerate_with_estimate(
                val_dl,
                f"Epoch {epoch_idx} training",
                start_idx=val_dl.num_workers,
            )
            '''

            batch_iter = enumerate(val_dl)

            for batch_idx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_idx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics
                )

            return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_idx, batch_tup, batch_size, metrics, classification_threshold=0.5):
        input_t, labels, series_list, slice_idx_list = batch_tup

        input_dev = input_t.to(device=self.device, non_blocking=True)
        labels_dev = labels.to(device=self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_dev, labels_dev = self.augmentation_model(input_dev, labels_dev)

        prediction_dev = self.segmentation_model(input_dev)

        dice_loss = self.dice_loss(prediction_dev, labels_dev)
        fn_loss = self.dice_loss(prediction_dev * labels_dev, labels_dev)

        start_idx = batch_idx * batch_size
        end_idx = start_idx + labels_dev.size(0)

        with torch.no_grad():
            predictions_bool = (prediction_dev[:, 0:1] > classification_threshold).to(torch.float32)

            tp = (predictions_bool * labels_dev).sum(dim=[1, 2, 3])
            fn = ((1 - predictions_bool) * labels_dev).sum(dim=[1, 2, 3])
            fp = (predictions_bool * (~labels_dev)).sum(dim=[1, 2, 3])

            metrics[METRICS_LOSS_IDX, start_idx:end_idx] = dice_loss
            metrics[METRICS_TP_IDX, start_idx:end_idx] = tp
            metrics[METRICS_FN_IDX, start_idx:end_idx] = fn
            metrics[METRICS_FP_IDX, start_idx:end_idx] = fp

        return dice_loss.mean() + fn_loss.mean() * 8

    def dice_loss(self, prediction_g, label_g, epsilon=1):
        dice_label_g = label_g.sum(dim=[1, 2, 3])
        dice_prediction_g = prediction_g.sum(dim=[1, 2, 3])
        dice_correct_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

        dice_ratio_g = (2 * dice_correct_g + epsilon) / (dice_prediction_g + dice_label_g + epsilon)

        return 1 - dice_ratio_g

    def log_images(self, epoch_idx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]

        for series_idx, series_uid in enumerate(images):
            ct = get_ct(series_uid)

            for slice_idx in range(6):
                ct_idx = slice_idx * (ct.hu_a.shape[0] - 1) // 5

                sample_tup = dl.dataset.getitem_full_slice(series_uid, ct_idx)

                ct_t, label_t, series_uid, ct_idx = sample_tup

                input_dev = ct_t.to(self.device).unsqueeze(0)
                label_dev = label_t.to(self.device).unsqueeze(0)

                prediction_dev = self.segmentation_model(input_dev)[0]
                prediction_a = prediction_dev.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_dev.to('cpu').numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ct_slice_a = ct_t[dl.dataset.context_slices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))

                # False Positive => Red
                image_a[:, :, 0] += np.logical_and(prediction_a, (1 - label_a))

                # False Negative => Orange
                image_a[:, :, 0] += np.logical_and((1 - prediction_a), label_a)
                image_a[:, :, 1] += np.logical_and(((1 - prediction_a), label_a)) * 0.5

                # True Positive => Green
                image_a[:, :, 1] += np.logical_and(prediction_a, label_a)

                # [0, 2] -> [0, 1] to represent float values
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_idx}_prediction_{slice_idx}',
                    image_a,
                    self.total_training_samples_count,
                    dataformats='HWC'
                )

                if epoch_idx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ct_slice_a.reshape((512, 512, 1))

                    image_a[:, :, 1] += label_a  # Green

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_idx,
                            slice_idx,
                        ),
                        image_a,
                        self.total_training_samples_count,
                        dataformats='HWC',
                    )

                # This flush prevents TB from getting confused about which data item belongs where.
                writer.flush()

    def log_metrics(self, epoch_idx, mode_str, metrics):
        log.info(f"E{epoch_idx} {type(self).__name__}")

        metrics_a = metrics.detach().numpy()

        sum_a = metrics_a.sum(axis=1)

        all_label_count = sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]

        metrics_dict = dict()

        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_IDX].mean()

        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_IDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_IDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_IDX] / (all_label_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_IDX] \
                                                   / ((sum_a[METRICS_TP_IDX] + sum_a[METRICS_FP_IDX]) or 1)
        recall = metrics_dict['pr/recall'] = sum_a[METRICS_TP_IDX] \
                                             / ((sum_a[METRICS_TP_IDX] + sum_a[METRICS_FN_IDX]) or 1)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(f"E{epoch_idx} {mode_str:10} {metrics_dict['loss/all']:.4f} loss, "
                 + f"{metrics_dict['pr/precision']:.4f}% precision"
                 + f"{metrics_dict['pr/recall']:.4f}% recall"
                 + f"{metrics_dict['pr/f1_score']:.4f}% f1_score")

        log.info(f"E{epoch_idx} {mode_str + '_all':10} {metrics_dict['loss/all']:.4f} loss, "
                 + f"{metrics_dict['percent_all/tp']:-5.1f}% TP"
                 + f"{metrics_dict['percent_all/fn']:-5.1f}% FN"
                 + f"{metrics_dict['percent_all/fp']:-9.1f}% FP")

        self.init_tensorboard_writers()
        writer = getattr(self, f"{mode_str}_writer")

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.total_training_samples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def save_model(self, type_str, epoch_idx, is_best=False):
        file_path = os.path.join(
            'models',
            f'{type_str}_{self.time_str}_{self.cli_args.comment}.{self.total_training_samples_count}.state')

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model

        if isinstance(model, nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_idx,
            'totalTrainingSamples_count': self.total_training_samples_count,
        }

        torch.save(state, file_path)

        if is_best:
            best_path = os.path.join(
                'models',
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')

            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


'''
class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--num-workers',
                            default=8,
                            help="Number of worker processes for background data loading.",
                            type=int
                            )

        parser.add_argument('--batch-size',
                            default=32,
                            help="Batch size to use for training.",
                            type=int
                            )

        parser.add_argument('--epochs',
                            default=1,
                            help="Number of epochs to train for.",
                            type=int
                            )

        parser.add_argument('--tb-prefix',
                            default='',
                            help="Data prefix to use for Tensorboard run.",
                            type=str
                            )

        parser.add_argument('--comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dwlpt',
                            type=str
                            )

        parser.add_argument('--balanced',
                            help="Balance the training data to half positive, half negative.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--flip',
                            help="Augment the training data by randomly flipping the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--offset',
                            help="Augment the training data by randomly offsetting the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--scale',
                            help="Augment the training data by randomly scaling the data.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--rotate',
                            help="Augment the training data by randomly rotating the data around X-Y axis.",
                            action='store_true',
                            default=False
                            )

        parser.add_argument('--noise',
                            help="Augment the training data by adding the random noise to the data.",
                            action='store_true',
                            default=False
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.augmentation_dict = {}

        if self.cli_args.augmented or self.cli_args.flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.noise:
            self.augmentation_dict['noise'] = 25.0

        self.train_writer = None
        self.val_writer = None
        self.total_training_samples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    def init_tensorboard_writers(self):
        if self.train_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def init_model(self):
        model = LunaModel()

        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices")

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            model.to(self.device)

        return model

    def init_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def init_train_dl(self):
        train_ds = LunaDataset(val_stride=10,
                               is_val_set_bool=False,
                               ratio_int=int(self.cli_args.balanced))

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda
                              )

        return train_dl

    def init_val_dl(self):
        val_ds = LunaDataset(val_stride=10, is_val_set_bool=True)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda
                            )

        return val_dl

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        for epoch_idx in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch_idx} of {self.cli_args.epochs}, "
                     + f"{len(train_dl)}/{len(val_dl)} batches of size "
                     + f"{self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}"
                     )

            train_metrics_t = self.do_training(epoch_idx, train_dl)
            self.log_metrics(epoch_idx, 'train', train_metrics_t)

            val_metrics_t = self.do_validation(epoch_idx, val_dl)
            self.log_metrics(epoch_idx, 'val', val_metrics_t)

    def do_training(self, epoch_idx, train_dl):
        self.model.train()

        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        batch_iter = enumerate_with_estimate(
            train_dl,
            f"Epoch {epoch_idx} training",
            start_idx=train_dl.num_workers,
        )

        batch_iter = enumerate(train_dl)

        for batch_idx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_idx,
                batch_tup,
                train_dl.batch_size,
                train_metrics
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_idx == 1 and batch_idx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.train_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.train_writer.close()

        self.total_training_samples_count += len(train_dl.dataset)

        return train_metrics.to('cpu')

    def do_validation(self, epoch_idx, val_dl):
        with torch.no_grad():
            self.model.eval()

            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device
            )

            batch_iter = enumerate_with_estimate(
                val_dl,
                f"Epoch {epoch_idx} training",
                start_idx=val_dl.num_workers,
            )

            batch_iter = enumerate(val_dl)

            for batch_idx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_idx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics
                )

            return val_metrics.to('cpu')

    def compute_batch_loss(self, batch_idx, batch_tup, batch_size, metrics):
        input, labels, _series_list, _center_list = batch_tup

        input_dev = input.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        logits, probabilities = self.model(input_dev)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits,
                       labels[:, 1])

        start_idx = batch_idx * batch_size
        end_idx = start_idx + labels.size(0)

        metrics[METRICS_LABEL_IDX, start_idx:end_idx] = labels[:, 1].detach()
        metrics[METRICS_PRED_IDX, start_idx:end_idx] = probabilities[:, 1].detach()
        metrics[METRICS_LOSS_IDX, start_idx:end_idx] = loss.detach()

        return loss.mean()

    def log_metrics(self, epoch_idx, mode_str, metrics, classification_threshold=0.5):
        log.info(f"E{epoch_idx} {type(self).__name__}")

        neg_label_mask = metrics[METRICS_LABEL_IDX] <= classification_threshold
        neg_pred_mask = metrics[METRICS_PRED_IDX] <= classification_threshold
        pos_label_mask = metrics[METRICS_LABEL_IDX] > classification_threshold
        pos_pred_mask = metrics[METRICS_PRED_IDX] > classification_threshold

        neg_count = int(neg_label_mask.sum())
        pos_count = int(pos_label_mask.sum())

        true_negative = neg_correct = int((neg_pred_mask & neg_label_mask).sum())
        true_positive = pos_correct = int((pos_pred_mask & pos_label_mask).sum())
        false_positive = neg_count - true_negative
        false_negative = pos_count - true_positive

        metrics_dict = dict()

        metrics_dict['loss/all'] = metrics[METRICS_LOSS_IDX, :].mean()
        metrics_dict['loss/neg'] = metrics[METRICS_LOSS_IDX, neg_label_mask].mean()
        metrics_dict['loss/pos'] = metrics[METRICS_LOSS_IDX, pos_label_mask].mean()

        metrics_dict['pr/precision'] = precision = true_positive / np.float32(true_positive + false_positive)
        metrics_dict['pr/recall'] = recall = true_positive / np.float32(true_positive + false_negative)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        metrics_dict['correct/all'] = (neg_correct + pos_correct) / np.float32(metrics.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        log.info(f"E{epoch_idx} {mode_str:10} {metrics_dict['loss/all']:.4f} loss, "
                 + f"{metrics_dict['correct/all']:-5.1f}% correct"
                 + f"{metrics_dict['pr/precision']:.4f}% precision"
                 + f"{metrics_dict['pr/recall']:.4f}% recall"
                 + f"{metrics_dict['pr/f1_score']:.4f}% f1_score")

        log.info(f"E{epoch_idx} {mode_str + '_neg':10} {metrics_dict['loss/neg']:.4f} loss, "
                 + f"{metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count})")

        log.info(f"E{epoch_idx} {mode_str + '_pos':10} {metrics_dict['loss/pos']:.4f} loss, "
                 + f"{metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count})")

        self.init_tensorboard_writers()
        writer = getattr(self, f"{mode_str}_writer")

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.total_training_samples_count)
'''

if __name__ == '__main__':
    SegmentationTrainingApp().main()
