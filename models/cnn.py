import argparse

import torch
import torchvision
from timm.models.efficientnet import efficientnet_b4
from torch import nn

from timm.models.regnet import regnety_040
from timm.models.resnet import resnet18, resnet50
from torch.nn import Conv2d
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST

from models import cal_weights_num
from utils.args import add_args


def resnet18_cifar10(args):
    print("Use resnet18")
    model = resnet18(num_classes=10, pretrained=False)
    return model


def _resnet50(args):
    print("Use resnet50")
    model = resnet50(num_classes=10, pretrained=False)
    if args.dataset_name == 'fmnist' and args.in_channels == 1:
        model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


def _efficient_b4(args):
    print("Use efficient_b4")
    model = efficientnet_b4(num_classes=10, pretrained=False)
    if args.dataset_name == 'fmnist' and args.in_channels == 1:
        model.conv_stem = Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model


def _regnety_4g(args):
    print("Use regnet4g")
    model = regnety_040(num_classes=10, pretrained=False)
    if args.dataset_name == 'fmnist' and args.in_channels == 1:
        model.stem.conv = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    return model


def simple_cnn_cifar10(args=None):
    print("Use simple cnn")
    model = CIFAR10_CNN()
    return model


class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Sequential(
                                   nn.Conv2d(3, 32, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv2 = nn.Sequential(
                                   nn.Conv2d(32, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(64, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Sequential(
                                nn.Linear(1024, 64),
                                nn.ReLU(),
                                )
        self.clf = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(self.dropout(x.flatten(1)))
        return self.clf(self.dropout(x))


if __name__ == '__main__':
    # model = resnet18_cifar10()
    # model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # model = regnety_4g(None)
    # model = _resnet50(None)
    # model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # print(model)
    #
    # data_dir = '/data/zxj/dataset/mnist'
    # transform = transforms.Compose([transforms.ToTensor()])
    # server_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    # dl = data.DataLoader(dataset=server_test_dataset, batch_size=2, shuffle=False, drop_last=False)
    # x = enumerate(dl)
    # x,y = x.__next__()
    # prob = model(y[0])
    # print(prob)
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    args.img_size = 224

    x = torch.randn(1, 1, args.img_size, args.img_size)
    # model = _resnet50(args)
    # model = _regnety_4g(args)
    model = _efficient_b4(args)
    n_weights = cal_weights_num(model)
    print(str(n_weights) + ' M')
