# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch


class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self.layer12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer19 = torch.nn.ReLU(inplace=True)
        self.layer20 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer23 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer24 = torch.nn.ReLU(inplace=True)
        self.layer25 = torch.nn.Dropout(p=0.5)
        self.layer26 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer27 = torch.nn.ReLU(inplace=True)
        self.layer28 = torch.nn.Dropout(p=0.5)
        self.layer29 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)

        self._initialize_weights()

    def forward(self, input0):
        out11 = input0.clone()
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = out20.size(0)
        out22 = out20.view(out21, -1)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        return out29

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)