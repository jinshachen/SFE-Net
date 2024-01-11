import torch
import torch.nn as nn
from unet import UNet


class LGD(nn.Module):
    def __init__(self, pretrained_path = "best_model_MoNuSeg256_BD.pth"):
        super(LGD, self).__init__()
        self.pretrained_path = pretrained_path
        self.model = UNet(in_channels=1, num_classes=2, base_c=32)
        
        if self.pretrained_path is not None:
            self.weights_dict = torch.load(self.pretrained_path, map_location=torch.device('cuda:0'))
            self.state_dict = self.weights_dict['model']
            self.model.load_state_dict(self.state_dict, strict=False)
            self.model.cuda()
        self.in_conv = self.model.in_conv
        self.down1 = self.model.down1
        self.down2 = self.model.down2
        self.down3 = self.model.down3
        self.down4 = self.model.down4
        self.up1 = self.model.up1
        self.up2 = self.model.up2
        self.up3 = self.model.up3
        self.up4 = self.model.up4
        self.out_conv = self.model.out_conv
        
    def forward(self, x):
        feat1 = self.in_conv(x)
        feat2 = self.down1(feat1)
        feat3 = self.down2(feat2)
        feat4 = self.down3(feat3)
        feat5 = self.down4(feat4)
        x1 = self.up1(feat5, feat4)
        x2 = self.up2(x1, feat3)
        x3 = self.up3(x2, feat2)
        x4 = self.up4(x3, feat1)

        # x1 = self.up1(feat5)
        # x2 = self.up2(x1)
        # x3 = self.up3(x2)
        # x4 = self.up4(x3)
        logits = self.out_conv(x4)
        return logits, feat1, feat2, feat3, feat4, feat5, x1, x2, x3, x4


if __name__ == "__main__":
    model = LGD()
