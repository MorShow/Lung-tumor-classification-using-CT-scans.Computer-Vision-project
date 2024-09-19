import random

import numpy as np
import torch
import torch.nn as nn

from utils.log import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def augment3d(input_t):
    transform_t = torch.eye(4, dtype=torch.float32)

    for i in range(3):
        if True:  # 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1
        if True:  # 'offset' in augmentation_dict:
            offset_float = 0.1
            random_float = (random.random() * 2 - 1)
            transform_t[3,i] = offset_float * random_float

    if True:
        angle_rad = random.random() * np.pi * 2
        s = np.sin(angle_rad)
        c = np.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        transform_t @= rotation_t

    affine_t = torch.nn.functional.affine_grid(
            transform_t[:3].unsqueeze(0).expand(input_t.size(0), -1, -1).cuda(),
            input_t.shape,
            align_corners=False,
        )

    augmented_chunk = torch.nn.functional.grid_sample(
            input_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )

    '''
    if False:  # 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t
    '''

    return augmented_chunk


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        return self.maxpool(out)


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)

        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

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

    def forward(self, batch):
        tail_out = self.tail_batchnorm(batch)

        block_out = self.block1(tail_out)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        out_flat = block_out.view(
            block_out.size(0), -1
        )

        tail_out = self.head_linear(out_flat)

        return tail_out, self.head_softmax(tail_out)
