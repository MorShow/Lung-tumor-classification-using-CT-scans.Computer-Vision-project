import random
import math

from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.unet import UNet
from utils.log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class UnetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        modules_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear
        }

        for m in self.modules():
            if type(m) in modules_set:
                nn.init.kaiming_normal_(m.weight.data,
                                        a=0,
                                        mode='fan_out',
                                        nonlinearity='relu')

                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / np.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        out_b = self.input_batchnorm(x)
        out_u = self.unet(out_b)
        out_f = self.final(out_u)
        return out_f


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2d_transformation_matrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)

        affine_t = F.affine_grid(transform_t[:, :2],
                                 input_g.size(),
                                 align_corners=False)

        augmented_input_g = F.grid_sample(input_g,
                                          affine_t,
                                          padding_mode='border',
                                          align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32),
                                          affine_t,
                                          padding_mode='border',
                                          align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2d_transformation_matrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = random.random() * 2 - 1
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = random.random() * 2 - 1
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([[c, -s, 0],
                                       [s, c, 0],
                                       [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t
