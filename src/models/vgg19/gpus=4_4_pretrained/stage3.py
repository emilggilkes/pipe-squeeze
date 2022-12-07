import torch


class Stage3(torch.nn.Module):
    def __init__(self, vgg19_pretrained):
        super(Stage3, self).__init__()
        self.layer12 = vgg19_pretrained.features[28]
        self.layer13 = vgg19_pretrained.features[29]
        self.layer14 = vgg19_pretrained.features[30]
        self.layer15 = vgg19_pretrained.features[31]
        self.layer16 = vgg19_pretrained.features[32]
        self.layer17 = vgg19_pretrained.features[33]
        self.layer18 = vgg19_pretrained.features[34]
        self.layer19 = vgg19_pretrained.features[35]
        self.layer20 = vgg19_pretrained.features[36]
        self.layer21 = vgg19_pretrained.avgpool
        self.layer24 = vgg19_pretrained.classifier[0]
        self.layer25 = vgg19_pretrained.classifier[1]
        self.layer26 = vgg19_pretrained.classifier[2]
        self.layer27 = vgg19_pretrained.classifier[3]
        self.layer28 = vgg19_pretrained.classifier[4]
        self.layer29 = vgg19_pretrained.classifier[5]
        self.layer30 = vgg19_pretrained.classifier[6]
        
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
        out21 = self.layer21(out20)
        out22 = out21.size(0)
        out23 = out21.view(out22, -1)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        
        return out30
