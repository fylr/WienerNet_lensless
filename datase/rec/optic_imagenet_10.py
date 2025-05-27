import os
from collections.abc import Sequence

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix, AutoAugmentPolicy, AutoAugment

from utils.image_loader import get_image_loader
from utils.transform import CustomCrop


# 准备数据，自定义DataSet结构体
class OImageNet10(Dataset):
    def __init__(self, ds_path, pair_dir: list[str] = None, kind='train', loader_type=None, transform=None,
                 target_transform=None):
        self._ds_path = ds_path
        if pair_dir is None:
            self.pair_dir = ['blur', 'gt']
        else:
            self.pair_dir = pair_dir
        self._kind = kind
        self.transform = transform
        self.target_transform = target_transform
        self.samples, self.targets = self._prepareData()
        self.image_loader = get_image_loader(loader_type=loader_type)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        sample_file_path = os.path.join(self._ds_path, self.pair_dir[0], self._kind, self.samples[idx])
        sample = self.image_loader(sample_file_path, chans=3)
        if self.transform is not None:
            sample = self.transform(sample)

        target_file_path = os.path.join(self._ds_path, self.pair_dir[1], self._kind, self.targets[idx])
        target = self.image_loader(target_file_path, chans=3)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    # 读取 image, label 的路径
    def _prepareData(self):
        if (not isinstance(self.pair_dir, Sequence)) or len(self.pair_dir) != 2:
            raise ValueError("pair_dir is a sequence, should have 2 values (default: ['blur', 'gt'])")

        samples_list = []
        targets_list = []
        if self._kind in ['train', 'val', 'test']:
            dir_name_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[0], self._kind))
            dir_name_list.sort()
            self.label_names = dir_name_list
            for dir_name in dir_name_list:
                file_name_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[0], self._kind, dir_name))
                file_name_list.sort()
                for file_name in file_name_list:
                    samples_list.append(os.path.join(dir_name, file_name))

            dir_name_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[1], self._kind))
            dir_name_list.sort()
            for dir_name in dir_name_list:
                file_name_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[1], self._kind, dir_name))
                file_name_list.sort()
                for file_name in file_name_list:
                    targets_list.append(os.path.join(dir_name, file_name))
        else:
            raise ValueError()

        return samples_list, targets_list


def create_transform(cfg, train=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = (480, 480)
    crop_box = None  # top, left, height, width
    crop_size = 480
    target_resize_size = (480, 480)
    # target_crop_box = (124, 126, 224, 224)  # [..., 124: 348, 126:350]
    # target_crop_box = (161, 163, 150, 150) # [..., 161: 311, 163: 313]
    # target_crop_box = (60, 60, 105, 105)  # center_crop=105
    target_crop_box = (152, 152, 176, 176)  # (top, left, height, width); center_crop=176
    target_crop_size = 224
    enable_resize = True

    interpolation = InterpolationMode(cfg.interpolation)

    test_trans_list = [transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
                       CustomCrop(crop_size, crop_box=crop_box, enable_resize=enable_resize,
                                  interpolation=interpolation)]
    target_trans_list = [transforms.Resize(target_resize_size, interpolation=interpolation, antialias=True),
                         CustomCrop(target_crop_size, crop_box=target_crop_box, enable_resize=enable_resize,
                                    interpolation=interpolation)]
    comm_trans_list = [transforms.ToTensor(), ]
    # comm_trans_list = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    test_trans_list.extend(comm_trans_list)
    test_transform = transforms.Compose(test_trans_list)
    target_trans_list.extend(comm_trans_list)
    target_transform = transforms.Compose(target_trans_list)

    # augment 在训练集上面做数据增强
    if train:
        # train_trans_list = [
        #     transforms.RandomResizedCrop(train_crop_size, scale=(0.75, 1.0), ratio=(0.75, 1.333),
        #                                  interpolation=interpolation, antialias=True)]
        train_trans_list = [transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
                            CustomCrop(crop_size, crop_box=crop_box, enable_resize=enable_resize,
                                       interpolation=interpolation)]
        hflip_prob = cfg.hflip_prob
        if hflip_prob > 0.0:
            train_trans_list.append(transforms.RandomHorizontalFlip(p=hflip_prob))
        auto_augment = getattr(cfg, 'auto_augment', None)
        if auto_augment is not None:
            if auto_augment == "ra":
                train_trans_list.append(RandAugment(magnitude=cfg.ra_magnitude, interpolation=interpolation))
            elif auto_augment == "ta_wide":
                train_trans_list.append(TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment == "augmix":
                train_trans_list.append(AugMix(severity=cfg.augmix_severity, interpolation=interpolation))
            else:
                aa_policy = AutoAugmentPolicy(auto_augment)
                train_trans_list.append(AutoAugment(policy=aa_policy, interpolation=interpolation))

        train_trans_list.extend(comm_trans_list)
        random_erase = cfg.random_erase
        if random_erase > 0.0:
            train_trans_list.append(transforms.RandomErasing(p=random_erase))
        train_transform = transforms.Compose(train_trans_list)

        return train_transform, test_transform, target_transform
    else:  # 不做数据增强，常规transform
        return test_transform, target_transform


# 获取数据
def create_dataset(cfg, train=True):
    train_transform, test_transform, target_transform = None, None, None
    if train:
        train_transform, test_transform, target_transform = create_transform(cfg, train=True)
    else:
        test_transform, target_transform = create_transform(cfg, train=False)

    # 训练集、验证集
    if train:
        dataset_train = OImageNet10(cfg.dataset_dir, pair_dir=['blur', 'gt'], kind="train",
                                    loader_type=cfg.image_loader_type,
                                    transform=train_transform, target_transform=target_transform)
        dataset_validation = OImageNet10(cfg.dataset_dir, pair_dir=['blur', 'gt'], kind="val",
                                         loader_type=cfg.image_loader_type,
                                         transform=train_transform, target_transform=target_transform)

        return dataset_train, dataset_validation
    else:
        # 测试集，暂时用验证集
        dataset_test = OImageNet10(cfg.dataset_dir, pair_dir=['blur', 'gt'], kind="val",
                                   loader_type=cfg.image_loader_type,
                                   transform=test_transform, target_transform=target_transform)

        # dataset_test = OImageNet10(cfg.dataset_dir, pair_dir=['blur', 'gt'], kind="test",
        #                            loader_type=cfg.image_loader_type,
        #                            transform=test_transform, target_transform=target_transform)

        return dataset_test


if __name__ == '__main__':
    from PIL import Image
    import torch
    import matplotlib.pyplot as plt

    train_trans_list = [transforms.Resize((224, 224), antialias=True),
                        CustomCrop(224, crop_box=(72, 85, 65, 65)),
                        transforms.ToTensor()]
    train_transform = transforms.Compose(train_trans_list)

    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/rebuild_mswn_unet/train/n02747177/n02747177_10088.tif"
    img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/rebuild_wiener_net/train/n02747177/n02747177_10088.tif"
    img = cv2.imread(img_blur_path, cv2.IMREAD_COLOR)
    img = Image.fromarray(img)
    img = train_transform(img)
    img = (img * 255).clip(min=0, max=255).type(torch.uint8).permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(img[..., ::-1]), plt.title('recons'), plt.axis('off'), plt.show()
    plt.close()
