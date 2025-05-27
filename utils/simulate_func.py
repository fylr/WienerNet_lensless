import datetime
import time
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from torchvision import transforms

from utils.image_loader import pil_loader

# InterpolationFlags = transforms.InterpolationMode.BICUBIC
# InterpolationFlags = transforms.InterpolationMode.BILINEAR
# InterpolationFlags = transforms.InterpolationMode.NEAREST_EXACT
InterpolationFlags = transforms.InterpolationMode.NEAREST

def crop(img, side_edge=None):
    """ img: (..., C, H, W)
        side_edge (tuple): (top, bottom, left, right)
    """
    if side_edge is None:
        top = img.shape[-2] // 4
        bottom = top + img.shape[-2] // 2
        left = img.shape[-1] // 4
        right = left + img.shape[-1] // 2
    else:
        top, bottom, left, right = side_edge

    return img[..., top:bottom, left:right]


def pad(img, pad=None):
    """ img: (..., C, H, W)
        pad (tuple): (padding_left, padding_right, padding_top, padding_bottom)
    """
    # 依照从后往前的顺序依照pad的值对input进行padding
    if pad is None:
        pad = (img.shape[-1] // 2, img.shape[-1] - img.shape[-1] // 2,
               img.shape[-2] // 2, img.shape[-2] - img.shape[-2] // 2)

    return F.pad(img, pad, 'constant', 0)


def resize(img, size=(224, 224)):
    """ img: (..., C, H, W)
        size (tuple): (H, W)
    """
    return transforms.Resize(size, InterpolationFlags, antialias=True)(img)


def img2fft(img):
    return fft.fftshift(fft.fft2(fft.ifftshift(img, dim=(-1, -2))), dim=(-1, -2))

    # return fft.fft2(fft.ifftshift(img))
    # return fft.fftshift(fft.fft2(img))


# 非相干成像，强度图是模长；相干成像，强度图是模长的平方
def fft2img(img_fft):
    return torch.real(fft.fftshift(fft.ifft2(fft.ifftshift(img_fft, dim=(-1, -2))), dim=(-1, -2)))
    # return torch.abs(fft.fftshift(fft.ifft2(fft.ifftshift(img_fft, dim=(-1, -2))), dim=(-1, -2)))
    # return torch.abs(fft.fftshift(fft.ifft2(fft.ifftshift(img_fft, dim=(-1, -2))), dim=(-1, -2))) ** 2

    # return torch.real(fft.fftshift(fft.ifft2(img_fft)))
    # return torch.abs(fft.ifftshift(fft.ifft2(img_fft)))


def H(img, h_fft):
    img_fft = img2fft(img)
    return fft2img(img_fft * h_fft)


def HT(img, h_fft_conj):
    img_fft = img2fft(img)
    return fft2img(img_fft * h_fft_conj)


def Psi(img):
    return torch.stack((torch.roll(img, 1, dims=-2) - img, torch.roll(img, 1, dims=-1) - img), dim=-1)


def PsiT(img):
    diff1 = torch.roll(img[..., 0], -1, dims=-2) - img[..., 0]
    diff2 = torch.roll(img[..., 1], -1, dims=-1) - img[..., 1]
    return diff1 + diff2


def precompute_PsiTPsi(img_size):
    PsiTPsi = torch.zeros(img_size, dtype=torch.float)
    PsiTPsi[0, 0] = 4.
    PsiTPsi[0, 1] = PsiTPsi[1, 0] = PsiTPsi[0, -1] = PsiTPsi[-1, 0] = -1.
    # PsiTPsi = torch.abs(fft.fft2(PsiTPsi))
    PsiTPsi = img2fft(PsiTPsi)
    return PsiTPsi


def SoftThresh(x, tau):
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.zeros_like(x))


def image_normalize(img, img_size=None, is_psf=False):
    if img_size is not None:
        # min_val, max_val = img.min(), img.max()
        img = transforms.Resize(img_size, interpolation=InterpolationFlags, antialias=True)(img)
        # img = img.clip(min=min_val, max=max_val)
    # (B x C x H x W)
    img_normalized = torch.zeros_like(img)
    for bi in range(img.shape[0]):
        if is_psf and torch.linalg.norm(img[bi]):
            # img_normalized[bi] = img[bi] / img[bi].sum()
            # img_normalized[bi] = img[bi] / torch.linalg.norm(img[bi])
            img_normalized[bi] = img[bi]
        else:
            img_normalized[bi] = img[bi]

    return img_normalized


def image_load(img_path, chans, img_size, is_psf=False, bg_pix=None):
    if not path.exists(img_path):
        raise FileExistsError(f"{img_path}")
    img = pil_loader(img_path, chans=chans)
    img = transforms.ToTensor()(img)  # (H x W x C) -> (C x H x W)
    # img = cv2.imread(img_path, flag)
    # img = transforms.ToTensor()(Image.fromarray(img[..., ::-1]))  # (H x W x C) -> (C x H x W)

    # 加载psf时可减去背景光
    if is_psf and bg_pix is not None:
        for ci in range(img.shape[0]):
            img[ci] -= img[ci, bg_pix[0]:bg_pix[1], bg_pix[0]:bg_pix[1]].mean()
        img = img.clip(min=0)

    # (B x C x H x W)
    img = image_normalize(img.unsqueeze(0), img_size, is_psf=is_psf)

    # img = transforms.functional.hflip(img)#测试图片翻转
    return img


# img (B x C x H x W)
def image_quantize(img, max_val=255):
    out_shape = img.shape
    img_flat = img.reshape(out_shape[0], -1)
    img_min = torch.minimum(img_flat.min(dim=-1)[0], torch.zeros(out_shape[0], device=img.device))
    img_max, _ = torch.max(img_flat - img_min.view(-1, 1), 1)
    img = (img - img_min.view(-1, 1, 1, 1)) / img_max.view(-1, 1, 1, 1)
    img = img * max_val

    return img


# 加泊松噪声
def add_shot_noise(image, snr_db=40):
    if image.min() < 0:
        image -= image.min()

    noise = torch.poisson(image)
    image_eng = torch.linalg.norm(image) ** 2
    noise_eng = torch.linalg.norm(noise) ** 2

    fact = image_eng / noise_eng / np.sqrt(10 ** (snr_db / 10))

    return image + fact * noise


# 加高斯噪声
def add_guass_noise(image, mean=0, std=0.01):
    noise = torch.normal(mean=mean, std=std, size=image.shape, device=image.device)
    return image + noise


if __name__ == '__main__':
    psf_path = "/home/chenky/datasets/psf/psf.tif"
    # psf_path = "/home/chenky/datasets/psf/waller_psf.tiff"
    img_src_path = "/home/chenky/datasets/Optic_Mnist/gt/1.tif"
    img_dst_path = "/home/chenky/datasets/Optic_Mnist/mask/1.tif"

    start_time = time.time()
    img_size = (360, 480)
    img = cv2.imread(psf_path, cv2.IMREAD_GRAYSCALE)
    img = transforms.ToTensor()(img)
    img = pad(img)
    img = (img * 255).clip(min=0, max=255).type(torch.uint8)
    img = img.squeeze().cpu().numpy()
    cv2.imwrite("/home/chenky/datasets/psf/pad_psf.tif", img)
    plt.imshow(img, cmap='gray'), plt.title(f'src'), plt.show(), plt.close()

    print(f'Total cost time: {datetime.timedelta(seconds=time.time() - start_time)}')
