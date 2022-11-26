# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from .vgg19 import VGG19Partitioned
from collections import OrderedDict

def arch():
    return "vgg19"

def model():
    names = ['stage0', 'stage1']
    stages = [
        Stage0(),
        Stage1(),
    ]
    return OrderedDict(zip(names,stages))

def full_model():
    return VGG19Partitioned()
