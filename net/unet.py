from torch import nn

from utils.simulate_func import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down_DoubleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down_DoubleConv, self).__init__()
        self.down = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Up_DoubleConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nn_upsample=False):
        super(Up_DoubleConv, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if nn_upsample:
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    # x, x_up: (..., H, W)
    def forward(self, x, x_up):
        x = self.up_conv(x)
        if x.shape[-2:] != x_up.shape[-2:]:
            diffH = x_up.shape[-2] - x.shape[-2]
            diffW = x_up.shape[-1] - x.shape[-1]
            x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = torch.cat([x_up, x], dim=1)
        x = self.conv(x)
        return x


class ThreeResConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride: int = 1):
        super(ThreeResConv, self).__init__()
        if mid_channels is None:
            mid_channels = (in_channels + out_channels) // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.downsample(x)
        out = self.relu(out)

        return out


class Down_ThreeResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_ThreeResConv, self).__init__()
        self.down = nn.MaxPool2d(2, 2)
        # self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv = ThreeResConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        # x = self.bn(x)
        # x = self.relu(x)
        x = self.conv(x)
        return x


class Up_ThreeResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_ThreeResConv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )
        # self.up = nn.PixelShuffle(2)
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='nearest', align_corners=None),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        # )
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.conv = ThreeResConv(out_channels * 2, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # x, x_up: (..., H, W)
    def forward(self, x, x_up):
        x = self.up(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # if x.shape[-2:] != x_up.shape[-2:]:
        #     diffH = x_up.shape[-2] - x.shape[-2]
        #     diffW = x_up.shape[-1] - x.shape[-1]
        #     x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = torch.cat([x_up, x], dim=1)
        x = self.conv(x)
        return x


# 通过填充保证输出图像和输入图像的大小相同
class UNet(nn.Module):
    def __init__(self, in_chans=3, nn_upsample=False):
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(in_chans, 24)

        self.down_double_conv1 = Down_DoubleConv(24, 64)
        self.down_double_conv2 = Down_DoubleConv(64, 128)
        self.down_double_conv3 = Down_DoubleConv(128, 256)
        self.down_double_conv4 = Down_DoubleConv(256, 512)

        self.up_double_conv4 = Up_DoubleConv(512, 256, nn_upsample)
        self.up_double_conv3 = Up_DoubleConv(256, 128, nn_upsample)
        self.up_double_conv2 = Up_DoubleConv(128, 64, nn_upsample)
        self.up_double_conv1 = Up_DoubleConv(64, 24, nn_upsample)

        self.out_conv = nn.Conv2d(24, in_chans, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.down_double_conv1(x1)
        x3 = self.down_double_conv2(x2)
        x4 = self.down_double_conv3(x3)
        x5 = self.down_double_conv4(x4)

        x4 = self.up_double_conv4(x5, x4)
        x3 = self.up_double_conv3(x4, x3)
        x2 = self.up_double_conv2(x3, x2)
        x1 = self.up_double_conv1(x2, x1)

        x = self.out_conv(x1)
        # x = crop(x, side_edge=(83, 138, 92, 137))
        # x = crop(x, side_edge=(72, 137, 85, 150))
        # x = crop(x, side_edge=(74, 136, 81, 143))

        return x


if __name__ == '__main__':
    # img_gt_path = "/home/chenky/datasets/Optic_cats_vs_dogs/gt/cat.12504.tif"
    # img_gt = image_load(img_gt_path, flag=cv2.IMREAD_COLOR, img_size=(224, 224))
    # img_recons = (img_gt * 255).permute(0, 2, 3, 1).squeeze().clip(min=0, max=255).type(torch.uint8).cpu().numpy()
    # plt.imshow(img_recons[..., ::-1]), plt.title('recons'), plt.axis('off'), plt.show()
    # plt.imshow(img_recons[75:145, 75:145, ::-1], interpolation='bilinear'), plt.title('recons_crop'), plt.axis('off')
    # plt.show(), plt.close()

    # img_blur_path = "/home/chenky/datasets/Optic_cats_vs_dogs/blur/cat.12504.tif"
    img_blur_path = "/home/chenky/datasets/Optic_cats_vs_dogs/blur/dog.25011.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_CelebA/blur/130414.png"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/train/n02747177/n02747177_10088.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/val/n02747177/ILSVRC2012_val_00007335.tif"
    # img_blur_path = "/home/chenky/datasets/Optic_ImageNet_10/blur/train/n03063599/n03063599_8.tif"

    img_size = (224, 224)
    device = torch.device("cuda:0")

    model = UNet(in_chans=3, nn_upsample=False).to(device)
    model.eval()
    # print(model)

    # 加载已训练好的模型
    model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_cats_dogs-unet.pth"
    # model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_celeba-unet.pth"
    # model_load_path = "/home/chenky/PycharmProjects/CI_for_reconstruction/data/model/optic_imagenet_10-unet.pth"
    model_state_dict = torch.load(model_load_path, map_location='cpu')
    model.load_state_dict(model_state_dict['model'])

    img_blur = image_load(img_blur_path, flag=cv2.IMREAD_COLOR, img_size=img_size).to(device)
    img_recons = model.forward(img_blur)

    img_recons = (img_recons * 255).clip(min=0, max=255).type(torch.uint8).permute(0, 2, 3, 1).squeeze().cpu().numpy()
    plt.imshow(img_recons), plt.title('recons'), plt.axis('off'), plt.show()
    # plt.imshow(img_recons[75:145, 75:145, ...]), plt.title('recons_crop'), plt.axis('off'), plt.show()
    # plt.imshow(img_recons[83:138, 92:137, ::-1]), plt.title('recons_crop'), plt.axis('off'), plt.show()
    plt.close()
