import torch


class Stage2(torch.nn.Module):
    def __init__(self, vgg19_pretrained):
        super(Stage2, self).__init__()
        self.layer3 = vgg19_pretrained.features[19]
        self.layer4 = vgg19_pretrained.features[20]
        self.layer5 = vgg19_pretrained.features[21]
        self.layer6 = vgg19_pretrained.features[22]
        self.layer7 = vgg19_pretrained.features[23]
        self.layer8 = vgg19_pretrained.features[24]
        self.layer9 = vgg19_pretrained.features[25]
        self.layer10 = vgg19_pretrained.features[26]
        self.layer11 = vgg19_pretrained.features[27]
        
    def forward(self, input0):
        out0 = input0.clone()
        out3 = self.layer3(out0)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)

        return out11
