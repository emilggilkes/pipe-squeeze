# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .vgg19 import VGG19Partitioned
from collections import OrderedDict
from torchvision import models

def arch():
    return "vgg19"

def model():
    names = ['stage0', 'stage1', 'stage2', 'stage3']
    vgg19_pretrained = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
    stages = [
        Stage0(vgg19_pretrained),
        Stage1(vgg19_pretrained),
        Stage2(vgg19_pretrained),
        Stage3(vgg19_pretrained)
    ]
    return OrderedDict(zip(names,stages))

def full_model():
    return VGG19Partitioned()
