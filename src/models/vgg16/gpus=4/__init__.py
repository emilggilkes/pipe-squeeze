# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .stage0 import Stage0
from .stage1 import Stage1
from .vgg16 import VGG16Partitioned

def arch():
    return "vgg16"

def model():
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"])
    ]

def full_model():
    return VGG16Partitioned()
