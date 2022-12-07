# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torchvision import models
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3

vgg19_pretrained = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)

class VGG19Partitioned(torch.nn.Module):
    def __init__(self):
        super(VGG19Partitioned, self).__init__()
        self.stage0 = Stage0(vgg19_pretrained)
        self.stage1 = Stage1(vgg19_pretrained)
        self.stage2 = Stage2(vgg19_pretrained)
        self.stage3 = Stage3(vgg19_pretrained)

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        return out3
