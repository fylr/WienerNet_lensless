from collections import OrderedDict
from itertools import chain
from os import path
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models

from net.pretrain_models import MyResNet50, MyResNet18


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = f'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key.replace('lin', '').replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int], use_dropout=True, requires_grad=False):
        super(LinLayers, self).__init__()
        lins = []
        lays = [nn.Dropout(), ] if (use_dropout) else []
        for chans in n_channels_list:
            lins.append(nn.Sequential(*lays, nn.Conv2d(chans, 1, 1, stride=1, padding=0, bias=False)))
        self += nn.ModuleList(lins)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer('mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters()):
            param.requires_grad = state

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), start=1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class AlexNet(BaseNet):
    def __init__(self, weights, requires_grad):
        super(AlexNet, self).__init__()
        if weights is None:  # 加载lpips发表时使用的torchvision=0.2.1时的预训练模型
            model_temp = models.alexnet()
            model_state_dict = torch.load(path.join(torch.hub.get_dir(), 'checkpoints', 'alexnet-owt-4df8aa71.pth'),
                                          map_location='cpu')
            model_temp.load_state_dict(model_state_dict)
            self.layers = model_temp.features
        else:
            self.layers = models.alexnet(weights=weights).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(requires_grad)


class SqueezeNet(BaseNet):
    def __init__(self, weights, requires_grad):
        super(SqueezeNet, self).__init__()
        if weights is None:  # 加载lpips发表时使用的torchvision=0.2.1时的预训练模型
            model_temp = models.squeezenet1_1()
            model_state_dict = torch.load(path.join(torch.hub.get_dir(), 'checkpoints', 'squeezenet1_1-f364aa15.pth'),
                                          map_location='cpu')
            model_temp.load_state_dict(model_state_dict)
            self.layers = model_temp.features
        else:
            self.layers = models.squeezenet1_1(weights=weights).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(requires_grad)


class VGG16(BaseNet):
    def __init__(self, weights, requires_grad):
        super(VGG16, self).__init__()
        if weights is None:  # 加载lpips发表时使用的torchvision=0.2.1时的预训练模型
            model_temp = models.vgg16()
            model_state_dict = torch.load(path.join(torch.hub.get_dir(), 'checkpoints', 'vgg16-397923af.pth'),
                                          map_location='cpu')
            model_temp.load_state_dict(model_state_dict)
            self.layers = model_temp.features
        else:
            self.layers = models.vgg16(weights=weights).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(requires_grad)


def get_network(net_type: str, weights, requires_grad=False):
    if net_type == 'alex':
        return AlexNet(weights=weights, requires_grad=requires_grad)
    elif net_type == 'squeeze':
        return SqueezeNet(weights=weights, requires_grad=requires_grad)
    elif net_type in ['vgg', 'vgg16']:
        return VGG16(weights=weights, requires_grad=requires_grad)
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        weights: The pretrained weights to use ("DEFAULT"|None). Default: None
    """

    def __init__(self, net_type: str = 'alex', weights=None, use_dropout=True,
                 requires_grad=False, normalize=False):
        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type, weights=weights, requires_grad=requires_grad)

        # linear layers
        self.lins = LinLayers(self.net.n_channels_list, use_dropout=use_dropout, requires_grad=False)
        self.lins.load_state_dict(get_state_dict(net_type))
        self.normalize = normalize
        self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            x, y = 2 * x - 1, 2 * y - 1
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean(dim=(2, 3), keepdim=True) for d, l in zip(diff, self.lins)]

        return torch.stack(res, dim=0).sum(dim=0)
        # return torch.sum(torch.cat(res, dim=0), dim=0, keepdim=True)


class FeaturesLoss(nn.Module):
    def __init__(self, session):
        super(FeaturesLoss, self).__init__()
        session.data.features_loss_weights = [1., 1., 1.]
        self.sdata = session.data
        if session.config.dataset in ['optic_imagenet_10']:
            layers = MyResNet50(3, 10)
            model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_imagenet_10_gt-my_resnet50.pth"
        elif session.config.dataset in ['optic_cats_dogs']:
            layers = MyResNet18(3, 2)
            model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cats_dogs_gt-my_resnet18.pth"
        elif session.config.dataset in ['optic_cifar100']:
            layers = MyResNet50(3, 100)
            model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_cifar100_gt-my_resnet50.pth"
        else:
            layers = MyResNet18(3, 2)
            model_load_path = "/home/chenky/PycharmProjects/Computational_Imaging/data/model/optic_celeba_gt-my_resnet18.pth"
        model_state_dict = torch.load(model_load_path, map_location='cpu')
        layers.load_state_dict(model_state_dict['model'])

        for param in layers.parameters():
            param.requires_grad = False
        if session.config.dataset in ['optic_imagenet_10', 'optic_cifar100']:
            layers_list = list(layers.resnet50.children())
        else:
            layers_list = list(layers.resnet18.children())

        self.feat1 = nn.Sequential(*layers_list[:3])
        self.feat2 = nn.Sequential(*layers_list[3:5])
        self.feat3 = nn.Sequential(*layers_list[5:6])
        self.feat4 = nn.Sequential(*layers_list[6:7])
        self.feat5 = nn.Sequential(*layers_list[7:8])
        self.loss_fn = nn.MSELoss()
        self.loss_weights = [1., 1., 1.]

    def feats_forward(self, x):
        feat0 = x
        feat1 = self.feat1(feat0)
        feat2 = self.feat2(feat1)
        feat3 = self.feat3(feat2)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)

        return [feat0, feat2, feat5]
        # return [feat0, feat1, feat2, feat3, feat4, feat5]

    def forward(self, pred_label: torch.Tensor, label: torch.Tensor):
        self.loss_weights = self.sdata.features_loss_weights
        pred_feats = self.feats_forward(pred_label)
        feats = self.feats_forward(label)

        losses = 0
        for idx in range(len(self.loss_weights)):
            losses += self.loss_weights[idx] * self.loss_fn(pred_feats[idx], feats[idx])
            # print(f"{idx} - {self.loss_fn(pred_feats[idx], feats[idx]):>.6f}")

        return losses / sum(self.loss_weights)


if __name__ == '__main__':
    from easydict import EasyDict

    # for net_type in ['alex', 'squeeze', 'vgg']:
    #     url = f'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/{net_type}.pth'
    #     old_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', progress=True)
    #     print(f"{net_type}")
    #     for key, val in old_state_dict.items():
    #         print(f"{key}: min={val.min():>.3f}, max={val.max():>.3f}, mean={val.mean():>.3f}, std={val.std():>.3f}")

    device = torch.device("cuda:3")

    a = torch.randint(0, 255, size=(20, 3, 224, 224), dtype=torch.float32).to(device) / 255.
    b = torch.randint(0, 255, size=(20, 3, 224, 224), dtype=torch.float32).to(device) / 255.

    torch.serialization.add_safe_globals([EasyDict])
    # loss_fn = LPIPS(net_type='alex', weights="DEFAULT", normalize=True).to(device)
    loss_fn = FeaturesLoss(EasyDict({'data': {'features_loss_weights': [1., 1., 1., 1., 1.]},
                                     'config': {'dataset': 'optic_celeba'}})).to(device)

    print(loss_fn(a, a * 1.00 + b * 0.00).mean())
    print(loss_fn(a, a * 0.99 + b * 0.01).mean())
    print(loss_fn(a, a * 0.95 + b * 0.05).mean())
    print(loss_fn(a, a * 0.90 + b * 0.10).mean())
    print(loss_fn(a, a * 0.85 + b * 0.15).mean())

    # from utils.ly_lpips import Lpips
    #
    # lf = Lpips(net='alex', device=device)

    # from lpips import LPIPS
    #
    # lf = LPIPS(net='alex').to(device)
    # print(lf(a, a * 1.00 + b * 0.00, normalize=True).mean())
    # print(lf(a, a * 0.75 + b * 0.25, normalize=True).mean())
    # print(lf(a, a * 0.50 + b * 0.50, normalize=True).mean())
    # print(lf(a, a * 0.25 + b * 0.75, normalize=True).mean())
    # print(lf(a, a * 0.00 + b * 1.00, normalize=True).mean())
