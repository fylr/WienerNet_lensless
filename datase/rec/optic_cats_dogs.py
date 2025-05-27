import os
from collections.abc import Sequence
from copy import copy

import torch
from torch.utils.data import random_split, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix, AutoAugmentPolicy, AutoAugment

from utils.image_loader import get_image_loader
from utils.transform import CustomCrop


# pair_dir=['blur', 'gt']
class OCatsDogs(Dataset):
    def __init__(self, ds_path, pair_dir: list[str] = None, loader_type=None, transform=None, target_transform=None):
        self._ds_path = ds_path
        if pair_dir is None:
            self.pair_dir = ['blur', 'gt']
        else:
            self.pair_dir = pair_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_loader = get_image_loader(loader_type=loader_type)
        self.samples, self.targets = self._prepareData()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        sample_file_path = os.path.join(self._ds_path, self.pair_dir[0], self.samples[idx])
        sample = self.image_loader(sample_file_path, chans=3)
        if self.transform is not None:
            sample = self.transform(sample)

        target_file_path = os.path.join(self._ds_path, self.pair_dir[1], self.targets[idx])
        target = self.image_loader(target_file_path, chans=3)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _prepareData(self):
        if (not isinstance(self.pair_dir, Sequence)) or len(self.pair_dir) != 2:
            raise ValueError("pair_dir is a sequence, should have 2 values (default: ['blur', 'gt'])")
        samples_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[0]))
        samples_list.sort()
        targets_list = os.listdir(os.path.join(self._ds_path, self.pair_dir[1]))
        targets_list.sort()

        return samples_list, targets_list


def create_transform(cfg, train=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = (480, 480)
    crop_box = None  # top, left, height, width
    crop_size = 480
    target_resize_size = (480, 480)
    # target_crop_box = None  # top, left, height, width
    # target_crop_box = (124, 126, 224, 224)  # [..., 124: 348, 126:350]
    # target_crop_box = (161, 163, 150, 150) # [..., 161: 311, 163: 313]
    # target_crop_box = (60, 60, 105, 105)  # center_crop=105
    target_crop_box = (152, 152, 176, 176)  # (top, left, height, width); center_crop=176
    # target_crop_box = (200, 200, 80, 80)
    # target_crop_box = (48, 48, 384, 384)
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

    cats_dogs_ds = OCatsDogs(ds_path=cfg.dataset_dir, pair_dir=['blur', 'gt'], loader_type=cfg.image_loader_type,
                             transform=test_transform, target_transform=target_transform)
    # 原训练集约含25K(23429)个样本，从原训练集中划分72%的样本作为训练集，15%的样本作为验证集，剩余的样本作为测试集
    train_ds_len, val_ds_len = int(len(cats_dogs_ds) * 0.72), int(len(cats_dogs_ds) * 0.15)
    test_ds_len = len(cats_dogs_ds) - train_ds_len - val_ds_len
    dataset_train, dataset_validation, dataset_test = random_split(cats_dogs_ds,
                                                                   [train_ds_len, val_ds_len, test_ds_len],
                                                                   torch.Generator().manual_seed(cfg.ds_split_seed))
    if train:
        dataset_train.dataset = copy(dataset_train.dataset)
        dataset_train.dataset.transform = train_transform
        return dataset_train, dataset_validation
    else:
        return dataset_test
