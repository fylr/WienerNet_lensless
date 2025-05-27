from torch import nn
from torchvision.models import resnet18, resnet50, alexnet


# 定义模型 ResNet18
class MyResNet18(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000):
        super(MyResNet18, self).__init__()
        self.resnet18 = resnet18(weights='DEFAULT')
        if in_chans != 3:
            self.resnet18.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # self.resnet18.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # for params in self.features.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        x = self.resnet18(x)
        return x


# 定义模型 ResNet50
class MyResNet50(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000):
        super(MyResNet50, self).__init__()
        self.resnet50 = resnet50(weights='DEFAULT')
        if in_chans != 3:
            self.resnet50.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if num_classes != 1000:
            self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # self.resnet50.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        # for params in self.resnet50.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        x = self.resnet50(x)
        return x


# 定义模型 AlexNet
class MyAlexNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.alexnet = alexnet(weights='DEFAULT')
        self.alexnet.features[0] = nn.Conv2d(in_chans, 64, kernel_size=11, stride=4, padding=2)
        self.alexnet.classifier[-1] = nn.Linear(4096, num_classes)
        # for params in self.alexnet.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        x = self.alexnet(x)
        return x

