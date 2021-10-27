import torch.nn as nn
from ops import ConvBlock, DownBlock, UpBlock
from utils import fft3, ifft3, to_complex


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(Unet, self).__init__()
        self.down1 = ConvBlock(in_channels, opt.ngf)
        self.down2 = DownBlock(opt.ngf, opt.ngf * 2)
        self.down3 = DownBlock(opt.ngf * 2, opt.ngf * 4)
        self.bridge = DownBlock(opt.ngf * 4, opt.ngf * 8)
        self.up3 = UpBlock(opt.ngf * 8, opt.ngf * 4)
        self.up2 = UpBlock(opt.ngf * 4, opt.ngf * 2)
        self.up1 = UpBlock(opt.ngf * 2, opt.ngf)
        self.last_conv = nn.Conv3d(opt.ngf, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        br = self.bridge(d3)
        u3 = self.up3(br, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        out = self.last_conv(u1)
        return out


class PhysicsModel(nn.Module):
    def __init__(self):
        super(PhysicsModel, self).__init__()

    def forward(self, x, dipole_kernel):
        qsm_k = fft3(to_complex(x))  # x(r) -> x(k)
        phase_k = qsm_k * to_complex(dipole_kernel, mode='tile')  # b(k) = x(k) * d(k)
        out = ifft3(phase_k)  # b(k) -> b(r)
        out = out[:, :, :, :, :, 0]  # real part
        return out
