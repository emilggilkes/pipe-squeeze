import torch


class Stage0(torch.nn.Module):
    def __init__(self, vgg19_pretrained):
        super(Stage0, self).__init__()
        self.layer2 = vgg19_pretrained.features[0]
        self.layer3 = vgg19_pretrained.features[1]
        self.layer4 = vgg19_pretrained.features[2]
        self.layer5 = vgg19_pretrained.features[3]
        self.layer6 = vgg19_pretrained.features[4]
        self.layer7 = vgg19_pretrained.features[5]
        self.layer8 = vgg19_pretrained.features[6]
        self.layer9 = vgg19_pretrained.features[7]
        self.layer10 = vgg19_pretrained.features[8]
        self.layer11 = vgg19_pretrained.features[9]
        
    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return out11