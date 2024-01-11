from PIL import Image
import numpy as np
import pywt
from torchvision.transforms import functional as F
from torch import nn
import torch
from CBAM import CABlock, SABlock, CBAMBlock


def dwtRGB(img):
    """
    img: <PIL.Image.Image image mode=RGB>
    LL: 近似系数   低频分量
    LH, HL, HH: 细节系数  高频分量
    """
    if img.mode == "L":
        cA, (cH, cV, cD) = pywt.dwt2(np.array(img), 'haar')
    else:
        # 对每个通道进行小波变换
        coeffs_r = pywt.dwt2(np.array(img)[:, :, 0], 'haar')
        coeffs_g = pywt.dwt2(np.array(img)[:, :, 1], 'haar')
        coeffs_b = pywt.dwt2(np.array(img)[:, :, 2], 'haar')

        # 提取每个通道的低频和高频分量
        cA_r, (cH_r, cV_r, cD_r) = coeffs_r
        cA_g, (cH_g, cV_g, cD_g) = coeffs_g
        cA_b, (cH_b, cV_b, cD_b) = coeffs_b

        # 将每个通道的低频和高频分量合并为一个矩阵
        cA = np.dstack((cA_r,cA_g,cA_b))
        cH = np.dstack((cH_r,cH_g,cH_b))
        cV = np.dstack((cV_r,cV_g,cV_b))
        cD = np.dstack((cD_r,cD_g,cD_b))

    cA = np.uint8((cA - np.min(cA)) / (np.max(cA) - np.min(cA)) * 255)
    cH = np.uint8((cH - np.min(cH)) / (np.max(cH) - np.min(cH)) * 255)
    cV = np.uint8((cV - np.min(cV)) / (np.max(cV) - np.min(cV)) * 255)
    cD = np.uint8((cD - np.min(cD)) / (np.max(cD) - np.min(cD)) * 255)

    LL = Image.fromarray(cA)
    LH = Image.fromarray(cH)
    HL = Image.fromarray(cV)
    HH = Image.fromarray(cD)

    LL = F.to_tensor(LL)
    LH = F.to_tensor(LH)
    HL = F.to_tensor(HL)
    HH = F.to_tensor(HH)

    return LL, LH, HL, HH


def idwtRGB(LL, LH, HL, HH):
    coeffs = np.array(LL), (np.array(LH), np.array(HL), np.array(HH))
    img = pywt.idwt2(coeffs, "haar")
    return img


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(
        in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class DWTCNN(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(DWTCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=5 // 2)
        self.sa = SABlock(kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
        self.IDWT = IWT()

    def forward(self, x):
        x_LL, x_HL, x_LH, x_HH = self.DWT(x)
        x = torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
        x = self.sa(x)
        x = self.IDWT(x)
        return x
    

class DWTConv(nn.Module):
    def __init__(self, num_channels=2, kernel_size=7):
        super(DWTConv, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.DWT = DWT()
    
    def forward(self, x):
        x = self.DWT(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x