from torch import nn
from torchvision.transforms import functional as tF

from net.pretrain_models import MyResNet50, MyResNet18
from net.unet import DoubleConv, Up_DoubleConv, Down_DoubleConv, Up_ThreeResConv
from utils.simulate_func import *


class Wiener(nn.Module):
    def __init__(self, psf_path, in_chans=1, img_size=(256, 256)):
        super(Wiener, self).__init__()
        if in_chans not in [1, 3]:
            raise ValueError()
        self.img_size = img_size
        self.register_buffer('psf_img', image_load(psf_path, chans=in_chans, img_size=img_size, is_psf=True))

        self.nsr = nn.Parameter(torch.tensor([1.] * 3).reshape(1, -1, 1, 1), requires_grad=True)
        self.intensity_weight = nn.Parameter(torch.tensor([1.] * 3).reshape(1, -1, 1, 1), requires_grad=True)

    def forward(self, x):
        psf_img_fft = img2fft(self.psf_img * self.intensity_weight / 1000)
        psf_img_fft_inv = psf_img_fft.conj_physical() / (psf_img_fft.abs() ** 2 + self.nsr / 1000)
        blur_img_fft = img2fft(x)
        rec_img = fft2img(psf_img_fft_inv * blur_img_fft)
        # rec_img = crop(rec_img)

        # rec_img = rec_img[..., 124: 348, 126:350]
        # rec_img = rec_img[..., 161: 311, 163: 313]
        rec_img = tF.center_crop(rec_img, 176)
        rec_img = resize(rec_img, size=(224, 224))
        rec_img = rec_img.clip(min=0) / rec_img.amax(dim=[1, 2, 3], keepdim=True)
        return rec_img

    def weight_constraint(self):
        self.nsr.data.clamp_(min=1e-8)
        self.intensity_weight.data.clamp_(min=1e-8)

    def extra_repr(self) -> str:
        extra_str = ["psf_img", "nsr", "intensity_weight"]
        return '\n'.join(extra_str)


class Wiener_psf(nn.Module):
    def __init__(self, psf_path, in_chans=1, img_size=(256, 256)):
        super(Wiener_psf, self).__init__()
        if in_chans not in [1, 3]:
            raise ValueError()
        self.img_size = img_size
        self.register_buffer('psf_img', image_load(psf_path, chans=in_chans, img_size=img_size, is_psf=True))

        self.mid_chans = 16
        self.ks = 53  # 53, 49
        self.psf_crt_conv = nn.Sequential(
            nn.Conv2d(in_chans, self.mid_chans, kernel_size=(self.ks, 1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_chans, self.mid_chans, kernel_size=(1, self.ks), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_chans, in_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
        )
        # self.relu = nn.ReLU(inplace=True)

        self.nsr = nn.Parameter(torch.tensor([1.] * 3).reshape(1, -1, 1, 1), requires_grad=True)

    def forward(self, x):
        crt_psf_img = self.psf_crt_conv(self.psf_img)
        # crt_psf_img = self.relu(crt_psf_img + self.psf_img)
        crt_psf_img_fft = img2fft(crt_psf_img / 1000)
        crt_psf_img_fft_inv = crt_psf_img_fft.conj_physical() / (crt_psf_img_fft.abs() ** 2 + self.nsr / 1000)
        blur_img_fft = img2fft(x)
        rec_img = fft2img(crt_psf_img_fft_inv * blur_img_fft)
        # rec_img = crop(rec_img)

        # rec_img = rec_img[..., 124: 348, 126:350]
        # rec_img = rec_img[..., 161: 311, 163: 313]
        rec_img = tF.center_crop(rec_img, 176)
        rec_img = resize(rec_img, size=(224, 224))
        rec_img = rec_img.clip(min=0) / rec_img.amax(dim=[1, 2, 3], keepdim=True)
        return rec_img

    def weight_constraint(self):
        self.nsr.data.clamp_(min=1e-8)

    def extra_repr(self) -> str:
        extra_str = ["psf_img", "nsr", ]
        return '\n'.join(extra_str)


class Wiener_rec(nn.Module):
    def __init__(self, psf_path, in_chans=1, img_size=(256, 256)):
        super(Wiener_rec, self).__init__()
        if in_chans not in [1, 3]:
            raise ValueError()
        self.img_size = img_size
        self.register_buffer('psf_img', image_load(psf_path, chans=in_chans, img_size=img_size, is_psf=True))

        self.nsr = nn.Parameter(torch.tensor([1.] * 3).reshape(1, -1, 1, 1), requires_grad=True)
        # self.rec_crt_conv = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        # self.rec_crt_conv = ThreeResConv(in_chans, in_chans, in_chans * 4)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 64)
        # self.up1 = Up_DoubleConv(64, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 32)
        # self.down2 = Down_DoubleConv(32, 64)
        # self.down3 = Down_DoubleConv(64, 128)
        # self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        # self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        # self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        self.in_conv = DoubleConv(in_chans, 16)
        self.down1 = Down_DoubleConv(16, 32)
        self.down2 = Down_DoubleConv(32, 64)
        self.down3 = Down_DoubleConv(64, 128)
        self.down4 = Down_DoubleConv(128, 256)
        self.up4 = Up_DoubleConv(256, 128, nn_upsample=False)
        self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

    def forward(self, x):
        psf_img_fft = img2fft(self.psf_img / 1000)
        psf_img_fft_inv = psf_img_fft.conj_physical() / (psf_img_fft.abs() ** 2 + self.nsr / 1000)
        blur_img_fft = img2fft(x)
        rec_img = fft2img(psf_img_fft_inv * blur_img_fft)
        # rec_img = crop(rec_img)

        # rec_img = rec_img[..., 124: 348, 126:350]
        rec_img = tF.center_crop(rec_img, 176)
        rec_img = resize(rec_img, size=(224, 224))
        # crt_rec_img = self.rec_crt_conv(rec_img)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x3 = self.up3(x4, x3)
        # x2 = self.up2(x3, x2)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        x1 = self.in_conv(rec_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)
        crt_rec_img = self.out_conv(x1)

        crt_rec_img = crt_rec_img.clip(min=0) / crt_rec_img.amax(dim=[1, 2, 3], keepdim=True)
        return crt_rec_img

    def weight_constraint(self):
        self.nsr.data.clamp_(min=1e-8)

    def extra_repr(self) -> str:
        extra_str = ["psf_img", "nsr"]
        return '\n'.join(extra_str)


class Wiener_psf_rec(nn.Module):
    def __init__(self, psf_path, in_chans=1, img_size=(256, 256)):
        super(Wiener_psf_rec, self).__init__()
        if in_chans not in [1, 3]:
            raise ValueError()
        self.img_size = img_size
        self.register_buffer('psf_img', image_load(psf_path, chans=in_chans, img_size=img_size, is_psf=True))

        self.mid_chans = 16
        self.ks = 53  # 53, 49
        self.psf_crt_conv = nn.Sequential(
            nn.Conv2d(in_chans, self.mid_chans, kernel_size=(self.ks, 1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_chans, self.mid_chans, kernel_size=(1, self.ks), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(self.mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_chans, in_chans, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
        )
        self.nsr = nn.Parameter(torch.tensor([1.] * 3).reshape(1, -1, 1, 1), requires_grad=True)

        # self.rec_crt_conv = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        # self.rec_crt_conv = nn.Sequential(
        #     nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_chans),
        #     nn.ReLU(inplace=True),
        # )
        # self.rec_crt_conv = ThreeResConv(in_chans, in_chans, in_chans * 4)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 64)
        # self.up1 = Up_DoubleConv(64, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 32)
        # self.down2 = Down_DoubleConv(32, 64)
        # self.down3 = Down_DoubleConv(64, 128)
        # self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        # self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        # self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        self.in_conv = DoubleConv(in_chans, 16)
        self.down1 = Down_DoubleConv(16, 32)
        self.down2 = Down_DoubleConv(32, 64)
        self.down3 = Down_DoubleConv(64, 128)
        self.down4 = Down_DoubleConv(128, 256)
        self.up4 = Up_DoubleConv(256, 128, nn_upsample=False)
        self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

    def forward(self, x):
        crt_psf_img = self.psf_crt_conv(self.psf_img)
        crt_psf_img_fft = img2fft(crt_psf_img / 1000)
        crt_psf_img_fft_inv = crt_psf_img_fft.conj_physical() / (crt_psf_img_fft.abs() ** 2 + self.nsr / 1000)
        blur_img_fft = img2fft(x)
        rec_img = fft2img(crt_psf_img_fft_inv * blur_img_fft)
        # rec_img = crop(rec_img)

        # rec_img = rec_img[..., 124: 348, 126:350]
        rec_img = tF.center_crop(rec_img, 176)
        rec_img = resize(rec_img, size=(224, 224))
        # rec_img = rec_img.clip(min=0) / rec_img.amax(dim=[1, 2, 3], keepdim=True)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x3 = self.up3(x4, x3)
        # x2 = self.up2(x3, x2)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        x1 = self.in_conv(rec_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)
        crt_rec_img = self.out_conv(x1)

        crt_rec_img = crt_rec_img.clip(min=0) / crt_rec_img.amax(dim=[1, 2, 3], keepdim=True)

        return crt_rec_img

    def weight_constraint(self):
        self.nsr.data.clamp_(min=1e-8)

    def extra_repr(self) -> str:
        extra_str = ["psf_img", "nsr"]
        return '\n'.join(extra_str)


class Wiener_ms_psf_rec(nn.Module):
    def __init__(self, psf_path, in_chans=3, img_size=(256, 256)):
        super(Wiener_ms_psf_rec, self).__init__()
        if in_chans not in [1, 3]:
            raise ValueError()
        self.img_size = img_size
        self.register_buffer('psf_img', image_load(psf_path, chans=in_chans, img_size=img_size, is_psf=True))
        self.num_layers = 3
        self.alpha = [1e-3 * 2.25 ** idx for idx in range(self.num_layers)]
        self.beta = [1e3 * alpha ** 2 for alpha in self.alpha]
        self.nsr = nn.Parameter(torch.tensor([1.] * self.num_layers), requires_grad=True)
        self.ks = 53  # 53, 49
        self.in_conv_blur = nn.Identity()
        # self.down_blur1 = nn.PixelUnshuffle(2)
        # self.down_blur2 = nn.PixelUnshuffle(2)
        self.down_blur1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_chans * 4, in_chans * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans * 4),
            nn.ReLU(inplace=True),
        )
        self.down_blur2 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_chans * 16, in_chans * 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans * 16),
            nn.ReLU(inplace=True),
        )
        self.in_conv_psf = self._make_psf_crt_conv(in_chans, in_chans, mid_chans=16, stride=1)
        # self.in_conv_psf = nn.Identity()
        # self.in_conv_psf = DoubleConv(in_chans, in_chans, in_chans * 6)
        # self.in_conv_psf = ThreeResConv(in_chans, in_chans, in_chans * 6)
        # self.down_psf1 = nn.PixelUnshuffle(2)
        # self.down_psf2 = nn.PixelUnshuffle(2)
        self.down_psf1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_chans * 4, in_chans * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans * 4),
            nn.ReLU(inplace=True),
        )
        self.down_psf2 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_chans * 16, in_chans * 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_chans * 16),
            nn.ReLU(inplace=True),
        )
        self.up_rec2 = Up_ThreeResConv(in_chans * 16, in_chans * 4)
        self.up_rec1 = Up_ThreeResConv(in_chans * 4, in_chans)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 64)
        # self.up1 = Up_DoubleConv(64, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 64)
        # self.down2 = Down_DoubleConv(64, 128)
        # self.up2 = Up_DoubleConv(128, 64, nn_upsample=False)
        # self.up1 = Up_DoubleConv(64, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        # self.in_conv = DoubleConv(in_chans, 16)
        # self.down1 = Down_DoubleConv(16, 32)
        # self.down2 = Down_DoubleConv(32, 64)
        # self.down3 = Down_DoubleConv(64, 128)
        # self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        # self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        # self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        # self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

        self.in_conv = DoubleConv(in_chans, 16)
        self.down1 = Down_DoubleConv(16, 32)
        self.down2 = Down_DoubleConv(32, 64)
        self.down3 = Down_DoubleConv(64, 128)
        self.down4 = Down_DoubleConv(128, 256)
        self.up4 = Up_DoubleConv(256, 128, nn_upsample=False)
        self.up3 = Up_DoubleConv(128, 64, nn_upsample=False)
        self.up2 = Up_DoubleConv(64, 32, nn_upsample=False)
        self.up1 = Up_DoubleConv(32, 16, nn_upsample=False)
        self.out_conv = nn.Conv2d(16, in_chans, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv_blur(x)
        x2 = self.down_blur1(x1)
        x3 = self.down_blur2(x2)

        psf1 = self.in_conv_psf(self.psf_img)
        psf2 = self.down_psf1(psf1)
        psf3 = self.down_psf2(psf2)

        x3 = self.wiener_deconv(x3, psf3, 2)
        x2 = self.wiener_deconv(x2, psf2, 1)
        x1 = self.wiener_deconv(x1, psf1, 0)

        x2 = self.up_rec2(x3, x2)
        x1 = self.up_rec1(x2, x1)

        rec_img = tF.center_crop(x1, 176)
        # rec_img = tF.center_crop(x1, 136)
        # rec_img = tF.center_crop(x1, 112)
        rec_img = resize(rec_img, size=(224, 224))
        # rec_img = rec_img.clip(min=0) / rec_img.amax(dim=[1, 2, 3], keepdim=True)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x2 = self.up2(x3, x2)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        # x1 = self.in_conv(rec_img)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x3 = self.up3(x4, x3)
        # x2 = self.up2(x3, x2)
        # x1 = self.up1(x2, x1)
        # crt_rec_img = self.out_conv(x1)

        x1 = self.in_conv(rec_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)
        crt_rec_img = self.out_conv(x1)

        crt_rec_img = crt_rec_img.clip(min=0) / crt_rec_img.amax(dim=[1, 2, 3], keepdim=True)

        return crt_rec_img

    def _make_psf_crt_conv(self, in_chans, out_chans, mid_chans=None, stride=1) -> nn.Sequential:
        layers = []
        if mid_chans is None:
            mid_chans = (in_chans + out_chans) // 2
        layers.extend([
            nn.Conv2d(in_chans, mid_chans, kernel_size=(self.ks, 1), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chans, mid_chans, kernel_size=(1, self.ks), stride=1, padding='same', bias=False),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chans, out_chans, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        ])
        self.ks = self.ks // 2
        if self.ks % 2 == 0:
            self.ks += 1

        return nn.Sequential(*layers)

    # 因为rec的有效区域占中间150，psf的有效区域占中间300，blur大小为480>300+150，不需要pad填充就能完成合理重建
    def wiener_deconv(self, blur_img, psf_img, idx):
        psf_img_fft = img2fft(psf_img * self.alpha[idx])
        psf_img_fft_inv = psf_img_fft.conj_physical() / (psf_img_fft.abs() ** 2 + self.nsr[idx] * self.beta[idx])
        blur_img_fft = img2fft(blur_img)
        rec_img = fft2img(psf_img_fft_inv * blur_img_fft)
        # rec_img = crop(rec_img)
        # rec_img = rec_img.clip(min=0) / rec_img.amax(dim=[1, 2, 3], keepdim=True)
        return rec_img

    def weight_constraint(self):
        self.nsr.data.clamp_(min=1e-8)

    def extra_repr(self) -> str:
        extra_str = ["psf_img", "nsr"]
        return '\n'.join(extra_str)


class CnnNet(nn.Module):
    def __init__(self, psf_path, img_size=(256, 256), in_chans=3, num_classes=1000):
        super(CnnNet, self).__init__()
        self.wiener_decoder = Wiener_ms_psf_rec(psf_path, in_chans=in_chans, img_size=img_size)
        # self.classifier = MyResNet18(in_chans, num_classes)
        self.classifier = MyResNet50(in_chans, num_classes)

        self.init_weights()
        # for params in self.wiener_decoder.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        x_rec = self.wiener_decoder(x)
        x = self.classifier(x_rec)

        return x

    def init_weights(self):
        # model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_cats_dogs-wiener_net.pth.多层"
        # model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_imagenet_10-wiener_net.pth.多层"
        # model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_celeba-wiener_net-0.pth"
        model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_cifar100-wiener_net.pth"
        model_state_dict = torch.load(model_load_path, map_location='cpu')
        self.wiener_decoder.load_state_dict(model_state_dict['model'])

        # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cats_dogs_gt-my_resnet18.pth"
        # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cats_dogs_rebuild-my_resnet18.pth.wiener"
        # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_gt-my_resnet50.pth"
        # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_rebuild-my_resnet50.pth.多层"
        # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_celeba_rebuild-my_resnet18-0.pth"
        model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cifar100_rebuild-my_resnet50.pth"
        model_state_dict = torch.load(model_load_path, map_location='cpu')
        self.classifier.load_state_dict(model_state_dict['model'])

    def weight_constraint(self):
        if hasattr(self.wiener_decoder, 'weight_constraint') and callable(self.wiener_decoder.weight_constraint):
            self.wiener_decoder.weight_constraint()


if __name__ == '__main__':
    psf_path = "/home/chenky/datasets/psf/canny_psf.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_cats_vs_dogs/blur/cat.12504.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_CelebA/blur/135321.png"
    # img_blur_path = "/home/chenky/datasets/Optic_CelebA/blur/064628.png"
    # img_blur_path = "/home/chenky/datasets/Optic_CelebA/blur/130414.png"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/train/n02747177/n02747177_118.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10_org/blur/train/n02747177/n02747177_10088.tif"
    img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/train/n03063599/n03063599_8.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10_org/blur/val/n02747177/ILSVRC2012_val_00007335.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/train/n02747177/n02747177_3.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10_org/blur/train/n02708093/n02708093_1001.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/test/n03938244/zhentou.tif"

    # img_size = (224, 224)
    img_size = (480, 480)
    device = torch.device("cuda:0")
    # model = CnnNet(psf_path, img_size=img_size, in_chans=3, num_classes=2).to(device)
    model = CnnNet(psf_path, img_size=img_size, in_chans=3, num_classes=10).to(device)
    model.eval()

    # 加载已训练好的模型
    # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cats_dogs_gt-my_resnet18.pth"
    # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/checkpoint/optic_cats_dogs_blur-cnn_net.pth"
    model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_blur-cnn_net.pth"
    model_state_dict = torch.load(model_load_path, map_location='cpu')
    # model.load_state_dict(model_state_dict['model'])

    img_blur = image_load(img_blur_path, chans=3, img_size=img_size).to(device)
    _, img_recons = model.forward(img_blur)

    img_recons = (img_recons * 255).clip(min=0, max=255).type(torch.uint8).permute(0, 2, 3, 1).squeeze().cpu().numpy()
    plt.imshow(img_recons), plt.title('recons'), plt.axis('off'), plt.show(), plt.close()

    # import matplotlib.pyplot as plt
    # plt.imshow(samples[0].permute(1, 2, 0).detach().cpu().numpy()), plt.title('recons'), plt.axis(
    #     'off'), plt.show(), plt.close()
