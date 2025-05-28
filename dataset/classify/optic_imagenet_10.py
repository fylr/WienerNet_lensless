import os

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix, AutoAugmentPolicy, AutoAugment
from torchvision.transforms.functional import InterpolationMode

from utils.image_loader import get_image_loader
from utils.transform import CustomCrop


# 准备数据，自定义DataSet结构体
class OImageNet10(Dataset):
    def __init__(self, ds_path, subdir, kind, loader_type=None, transform=None):
        self._ds_path = ds_path
        self._subdir = subdir
        self._kind = kind
        self.transform = transform
        self.images, self.labels = self._prepareData()
        self.image_loader = get_image_loader(loader_type=loader_type)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        label = self.labels[idx]

        img_path = os.path.join(self._ds_path, self._subdir, self._kind, self.images[idx])
        image = self.image_loader(img_path, chans=3)
        # image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 读取 image, label 的路径
    def _prepareData(self):
        image_list = []
        label_list = []

        if self._kind in ["train", "val", 'test']:
            dir_name_list = os.listdir(os.path.join(self._ds_path, self._subdir, self._kind))
            dir_name_list.sort()
            self.label_names = dir_name_list
            for idx, dir_name in enumerate(dir_name_list):
                file_name_list = os.listdir(os.path.join(self._ds_path, self._subdir, self._kind, dir_name))
                file_name_list.sort()
                for file_name in file_name_list:
                    image_list.append(os.path.join(dir_name, file_name))
                    label_list.append(idx)
        else:
            raise ValueError()

        return image_list, label_list


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
    if cfg.dataset in ['optic_imagenet_10_gt']:
        resize_size = (480, 480)
        # target_crop_size = 224
        # crop_box = (124, 126, 224, 224)
        # crop_box = (161, 163, 150, 150)
        # crop_box = (60, 60, 105, 105)  # center_crop=105
        crop_box = (152, 152, 176, 176)  # center_crop=176
        crop_size = 224
        enable_resize = True
    elif cfg.dataset in ['optic_imagenet_10_rebuild']:
        resize_size = (224, 224)
        crop_size = 224
    elif cfg.dataset in ['optic_imagenet_10_blur']:
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
    comm_trans_list = [transforms.ToTensor(), ]
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
def create_dataset(cfg, train=True):
    train_transform, test_transform, target_transform = None, None, None
    if train:
        train_transform, test_transform, target_transform = create_transform(cfg, train=True)
    else:
        test_transform, target_transform = create_transform(cfg, train=False)

    if cfg.dataset == "optic_imagenet_10_gt":
        subdir = "gt"
    elif cfg.dataset == "optic_imagenet_10_blur":
        subdir = "blur"
        # subdir = "blur_lbp_default"
    elif cfg.dataset == "optic_imagenet_10_rebuild":
        # subdir = 'rebuild'
        # subdir = 'rebuild_admm'
        # subdir = 'rebuild_unet'
        subdir = 'rebuild_wiener_net'
        # subdir = 'rebuild_mswn_unet'
    else:
        subdir = "gt"

    # 训练集、验证集
    if train:
        dataset_train = OImageNet10(cfg.dataset_dir, subdir=subdir, kind="train",
                                    loader_type=cfg.image_loader_type, transform=train_transform)
        dataset_validation = OImageNet10(cfg.dataset_dir, subdir=subdir, kind="val",
                                         loader_type=cfg.image_loader_type, transform=test_transform)

        return dataset_train, dataset_validation
    # 测试集
    else:
        # 将验证集作为测试集，仅供临时测试
        dataset_test = OImageNet10(cfg.dataset_dir, subdir=subdir, kind="val",
                                   loader_type=cfg.image_loader_type, transform=test_transform)

        return dataset_test


if __name__ == '__main__':
    # import numpy as np
    #
    psf_path = "/home/chenky/datasets/psf/canny_psf.tif"
    ds_path = "/home/chenky/datasets/Optic_ImageNet_10/"
    #
    # dataset_train = OImageNet10(ds_path, subdir='raw', kind='train', transform=transforms.ToTensor())
    # dataset_val = OImageNet10(ds_path, subdir='raw', kind='val', transform=transforms.ToTensor())
    # img_shape_h_list, img_shape_w_list = [], []
    # for img, _ in dataset_val:
    #     img_shape_h_list.append(img.shape[1])
    #     img_shape_w_list.append(img.shape[2])
    # for img, _ in dataset_train:
    #     img_shape_h_list.append(img.shape[1])
    #     img_shape_w_list.append(img.shape[2])
    #
    # img_shape_h_list, img_shape_w_list = np.array(img_shape_h_list), np.array(img_shape_w_list)
    # print(f'img_shape_h_min: {img_shape_h_list.min()}, img_shape_w_min: {img_shape_w_list.min()}\n'
    #       f'img_shape_h_mean: {img_shape_h_list.mean()}, img_shape_w_mean: {img_shape_w_list.mean()}\n'
    #       f'img_shape_h_max: {img_shape_h_list.max()}, img_shape_w_max: {img_shape_w_list.max()}\n')

    from net.classify.cnn_net import CnnNet
    import torch
    from easydict import EasyDict
    from net.pretrain_models import MyResNet50

    device = torch.device("cuda:3")
    torch.serialization.add_safe_globals([EasyDict])

    # model = MyResNet50(in_chans=3, num_classes=10).to(device)
    # model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_rebuild-my_resnet50.pth.多层"
    # my_trans = transforms.Compose([transforms.ToTensor()])
    # dataset_test = OImageNet10(ds_path, subdir='rebuild_wiener_net', kind="test", loader_type='pil', transform=my_trans)

    img_size = (480, 480)
    model = CnnNet(psf_path, img_size=img_size, in_chans=3, num_classes=10).to(device)
    model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_blur-cnn_net.pth.多层"
    my_trans = transforms.Compose([
        CustomCrop(480, crop_box=None, enable_resize=True, interpolation=InterpolationMode('nearest'))
        , transforms.ToTensor()])
    dataset_test = OImageNet10(ds_path, subdir='blur', kind="test", loader_type='pil', transform=my_trans)

    model_state_dict = torch.load(model_load_path, map_location='cpu')
    model.load_state_dict(model_state_dict['model'])
    model.eval()

    idx = 0
    with torch.inference_mode():
        for samples, labels in dataset_test:
            samples = samples.to(device, non_blocking=True).unsqueeze(0)
            # labels = labels.to(device, non_blocking=True)

            outputs = model(samples)

            pt_outputs = torch.nn.Softmax()(outputs).cpu().numpy().squeeze()
            pred_labels = pt_outputs.argmax()
            print(f"labels={labels}, pred_labels={pred_labels}, "
                  f"name={dataset_test.images[idx]}, pt_outputs={pt_outputs}")
            idx += 1
