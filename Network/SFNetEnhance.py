import torch
import torch.nn as nn
import torch.nn.functional as F

from CBAM import CBAMBlock
from DwtFeat import DWT
from BaseNet import ModBlock


class TwoConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(TwoConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),  
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)    # 1x1卷积：分割头
        )


class SFNet(nn.Module):
    def __init__(self, in_channels=3, base_c=32, num_classes=2, bilinear: bool = True):
        super(SFNet, self).__init__()
        
        self.in_conv = DoubleConv(in_channels, base_c)   # UNet

        factor = 2 if bilinear else 1
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.twoconv1 = TwoConv(base_c, base_c * 2)
        self.twoconv2 = TwoConv(base_c * 2, base_c * 4)
        self.twoconv3 = TwoConv(base_c * 4, base_c * 8)
        self.twoconv4 = TwoConv(base_c * 8, base_c * 16 // factor)

        self.conv_down1 = nn.Conv2d(base_c, base_c * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_down2 = nn.Conv2d(base_c * 2, base_c * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_down3 = nn.Conv2d(base_c * 4, base_c * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_down4 = nn.Conv2d(base_c * 8, base_c * 16 // factor, kernel_size=3, stride=2, padding=1, bias=False)
    
        self.cbam0 = CBAMBlock(channel=base_c, reduction=16, kernel_size=7)
        self.cbam1 = CBAMBlock(channel=base_c * 2, reduction=16, kernel_size=7)
        self.cbam2 = CBAMBlock(channel=base_c * 4, reduction=16, kernel_size=5)
        self.cbam3 = CBAMBlock(channel=base_c * 8, reduction=16, kernel_size=3)
        self.cbam4 = CBAMBlock(channel=base_c * 16 // factor, reduction=16, kernel_size=3)

        self.mod1 = ModBlock(base_c)
        self.mod2 = ModBlock(base_c * 2)
        self.mod3 = ModBlock(base_c * 4)
        self.mod4 = ModBlock(base_c * 8)
        self.mod5 = ModBlock(base_c * 16 // factor)

        self.fusiondown1 = DoubleConv(base_c, base_c)
        self.fusiondown2 = DoubleConv(base_c * 2, base_c * 2)
        self.fusiondown3 = DoubleConv(base_c * 4, base_c * 4)
        self.fusiondown4 = DoubleConv(base_c * 8, base_c * 8)
        self.fusiondown5 = DoubleConv(base_c * 16 // factor, base_c * 16 // factor)

        # decoder
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        self.dwt = DWT()

    def forward(self, x: torch.Tensor):
        feat1 = self.in_conv(x)
        feat2 = self.down1(feat1)
        feat2 = self.cbam1(feat2)

        feat3 = self.down2(feat2)
        feat3 = self.cbam2(feat3)

        feat4 = self.down3(feat3)
        feat4 = self.cbam3(feat4)

        feat5 = self.down4(feat4)
        feat5 = self.cbam4(feat5)

        # dwt
        dwtf1 = self.in_conv(x)
        dwtf1 = self.cbam0(dwtf1)
        
        x_LL, x_HL, x_LH, x_HH = self.dwt(dwtf1)
        dwtf2 = torch.add(torch.add(torch.add(x_LL, x_HL), x_LH), x_HH)
        dwtf2 = self.twoconv1(dwtf2)
        dwtf2 = self.cbam1(dwtf2)
        
        x_LL, x_HL, x_LH, x_HH = self.dwt(dwtf2)
        dwtf3 = torch.add(torch.add(torch.add(x_LL, x_HL), x_LH), x_HH)
        dwtf3 = self.twoconv2(dwtf3)
        dwtf3 = self.cbam2(dwtf3)
        
        x_LL, x_HL, x_LH, x_HH = self.dwt(dwtf3)
        dwtf4 = torch.add(torch.add(torch.add(x_LL, x_HL), x_LH), x_HH)
        dwtf4 = self.twoconv3(dwtf4)
        dwtf4 = self.cbam3(dwtf4)
        
        x_LL, x_HL, x_LH, x_HH = self.dwt(dwtf4)
        dwtf5 = torch.add(torch.add(torch.add(x_LL, x_HL), x_LH), x_HH)
        dwtf5 = self.twoconv4(dwtf5) 
        dwtf5 = self.cbam4(dwtf5)

        feat1 = self.mod1(feat1, dwtf1)
        feat1 = self.fusiondown1(feat1)
        feat2 = self.mod2(feat2, dwtf2)
        feat2 = self.fusiondown2(feat2)
        feat3 = self.mod3(feat3, dwtf3)
        feat3 = self.fusiondown3(feat3)
        feat4 = self.mod4(feat4, dwtf4)
        feat4 = self.fusiondown4(feat4)
        feat5 = self.mod5(feat5, dwtf5)
        feat5 = self.fusiondown5(feat5)

        x1 = self.up1(feat5, feat4)
        x2 = self.up2(x1, feat3)
        x3 = self.up3(x2, feat2)
        x4 = self.up4(x3, feat1)

        logits = self.out_conv(x4)
        return logits, feat1, feat2, feat3, feat4, feat5, x1, x2, x3, x4