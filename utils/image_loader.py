from typing import Any, Callable

import accimage
import cv2
from PIL import Image


# 加载速度 accimag > pil > opencv, 而图片格式中 jpg > tif > png
# 使用accimage时，部分transforms中的操作不支持，所以默认使用PIL加载
def get_image_loader(loader_type: str = None) -> Callable:
    if loader_type != 'opencv':
        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy('file_system')

    if loader_type == "accimage":
        return accimage_loader
    elif loader_type == "pil":
        return pil_loader
    elif loader_type == "opencv":
        return opencv_loader
    else:
        return pil_loader


# accimage只对彩色jpg格式的图片有效，其他情况则使用pil
def accimage_loader(path: str, chans: int = 3) -> Any:
    if chans == 1:
        return pil_loader(path, chans)
    elif chans == 3:
        try:
            return accimage.Image(path)
        except OSError:
            # Potentially a decoding problem, fall back to PIL.Image
            return pil_loader(path)
    else:
        raise ValueError()


def pil_loader(path: str, chans: int = 3) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        if chans == 1:
            return img.convert("L")
        elif chans == 3:
            return img.convert("RGB")
        else:
            raise ValueError()


# def pil_loader_new(path: str, chans: int = 3) -> Image.Image:
#     img = Image.open(path)
#     if chans == 1:
#         return img.convert("L")
#     elif chans == 3:
#         return img.convert("RGB")
#     else:
#         raise ValueError()


def opencv_loader(path: str, chans: int = 3) -> Image.Image:
    if chans == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(img)
    elif chans == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return Image.fromarray(img[..., ::-1])
    else:
        raise ValueError()


if __name__ == '__main__':
    pass
