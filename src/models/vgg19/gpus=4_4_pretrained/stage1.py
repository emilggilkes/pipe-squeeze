import torch


class Stage1(torch.nn.Module):
    def __init__(self, vgg19_pretrained):
        super(Stage1, self).__init__()
        self.layer12 = vgg19_pretrained.features[10]
        self.layer13 = vgg19_pretrained.features[11]
        self.layer14 = vgg19_pretrained.features[12]
        self.layer15 = vgg19_pretrained.features[13]
        self.layer16 = vgg19_pretrained.features[14]
        self.layer17 = vgg19_pretrained.features[15]
        self.layer18 = vgg19_pretrained.features[16]
        self.layer19 = vgg19_pretrained.features[17]
        self.layer20 = vgg19_pretrained.features[18]

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
        return out20
