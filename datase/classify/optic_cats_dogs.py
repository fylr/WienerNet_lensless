import os
from copy import copy

import torch
from torchvision import transforms
from torch.utils.data import random_split, Dataset
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix, AutoAugmentPolicy, AutoAugment
from torchvision.transforms.functional import InterpolationMode

from utils.image_loader import get_image_loader
from utils.transform import CustomCrop


class OCatsDogs(Dataset):
    def __init__(self, ds_path: str, subdir, loader_type=None, transform=None,
                 target_transform=None, targetdir='gt'):
        self._ds_path = ds_path
        self._subdir = subdir
        self.image_loader = get_image_loader(loader_type=loader_type)
        self.transform = transform
        self.target_transform = target_transform
        self.targetdir = targetdir
        self.samples, self.labels, self.targets = self._prepareData()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        label = self.labels[idx]

        sample_file_path = os.path.join(self._ds_path, self._subdir, self.samples[idx])
        sample = self.image_loader(sample_file_path, chans=3)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

        # if self.target_transform is None:
        #     target = label
        # else:
        #     target_file_path = os.path.join(self._ds_path, self.targetdir, self.targets[idx])
        #     target = self.image_loader(target_file_path, chans=3)
        #     target = self.target_transform(target)
        #
        # return sample, label, target

    def _prepareData(self):
        samples_list = os.listdir(os.path.join(self._ds_path, self._subdir))
        samples_list.sort()
        targets_list = os.listdir(os.path.join(self._ds_path, self.targetdir))
        targets_list.sort()
        # label {'cats': 0, 'dogs': 1}
        self.label_names = ['cats', 'dogs']
        label_list = []
        for file_name in samples_list:
            if file_name.startswith('cat.'):
                label_list.append(0)
            elif file_name.startswith('dog.'):
                label_list.append(1)

        return samples_list, label_list, targets_list


def create_transform(cfg, train=True, with_target=False):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = (224, 224)
    crop_size = 224
    crop_box = None  # top, left, height, width
    enable_resize = True
    target_resize_size = (224, 224)
    target_crop_size = 224
    target_crop_box = None  # top, left, height, width
    if cfg.dataset in ['optic_cats_dogs_gt']:
        resize_size = (480, 480)
        # target_crop_size = 224
        # crop_box = (124, 126, 224, 224)
        # crop_box = (161, 163, 150, 150)
        # crop_box = (60, 60, 105, 105)  # center_crop=105
        crop_box = (152, 152, 176, 176)  # center_crop=176
        crop_size = 224
        enable_resize = True
    elif cfg.dataset in ['optic_cats_dogs_rebuild']:
        resize_size = (224, 224)
        crop_size = 224
    elif cfg.dataset in ['optic_cats_dogs_blur']:
        resize_size = (480, 480)
        # crop_box = (124, 126, 224, 224)
        crop_size = 480
        target_resize_size = (480, 480)
        # target_crop_box = (124, 126, 224, 224)
        # target_crop_box = (60, 60, 105, 105)  # center_crop=105
        target_crop_box = (152, 152, 176, 176)  # center_crop=176
        target_crop_size = 224
        enable_resize = True

    interpolation = InterpolationMode(cfg.interpolation)
    # 224x224的带填充rebuild图像，crop出有效区域
    test_trans_list = [transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
                       CustomCrop(crop_size, crop_box=crop_box, enable_resize=enable_resize,
                                  interpolation=interpolation)]
    comm_trans_list = [transforms.ToTensor()]
    # comm_trans_list = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    test_trans_list.extend(comm_trans_list)
    test_transform = transforms.Compose(test_trans_list)

    if with_target:
        target_trans_list = [transforms.Resize(target_resize_size, interpolation=interpolation, antialias=True),
                             CustomCrop(target_crop_size, crop_box=target_crop_box, enable_resize=enable_resize,
                                        interpolation=interpolation)]
        target_trans_list.extend(comm_trans_list)
        target_transform = transforms.Compose(target_trans_list)
    else:
        target_transform = None

    # augment 在训练集上面做数据增强
    if train:
        # train_trans_list = [
        #     transforms.RandomResizedCrop(crop_size, scale=(0.75, 1.0), ratio=(0.75, 1.333),
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
def create_dataset(cfg, train=True, with_target=False):
    train_transform, test_transform, target_transform = None, None, None
    if train:
        train_transform, test_transform, target_transform = create_transform(cfg, train=True, with_target=with_target)
    else:
        test_transform, target_transform = create_transform(cfg, train=False, with_target=with_target)

    if cfg.dataset == "optic_cats_dogs_gt":
        subdir = "gt"
    elif cfg.dataset == "optic_cats_dogs_blur":
        subdir = "blur"
        # subdir = 'blur_lbp_default'
        # subdir = 'blur_lbp_ror'
        # subdir = 'blur_lbp_uniform'
    elif cfg.dataset == "optic_cats_dogs_rebuild":
        # subdir = "rebuild_admm"
        subdir = "rebuild_wiener_net"
        # subdir = "rebuild_unet"
        # subdir = "rebuild_mswn_unet"
    else:
        subdir = "gt"

    if with_target:
        cats_dogs_ds = OCatsDogs(ds_path=cfg.dataset_dir, subdir=subdir,
                                 loader_type=cfg.image_loader_type, transform=test_transform,
                                 target_transform=target_transform, targetdir='gt')
    else:
        cats_dogs_ds = OCatsDogs(ds_path=cfg.dataset_dir, subdir=subdir,
                                 loader_type=cfg.image_loader_type, transform=test_transform)
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


if __name__ == '__main__':
    import numpy as np

    cats_dogs_ds = OCatsDogs(ds_path="/home/chenky/datasets/Optic_cats_vs_dogs/", subdir="raw",
                             loader_type="pil", transform=transforms.ToTensor())
    img_shape_h_list, img_shape_w_list = [], []
    for img, _ in cats_dogs_ds:
        img_shape_h_list.append(img.shape[1])
        img_shape_w_list.append(img.shape[2])

    img_shape_h_list, img_shape_w_list = np.array(img_shape_h_list), np.array(img_shape_w_list)
    print(f'img_shape_h_min: {img_shape_h_list.min()}, img_shape_w_min: {img_shape_w_list.min()}\n'
          f'img_shape_h_mean: {img_shape_h_list.mean()}, img_shape_w_mean: {img_shape_w_list.mean()}\n'
          f'img_shape_h_max: {img_shape_h_list.max()}, img_shape_w_max: {img_shape_w_list.max()}\n')
