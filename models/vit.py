import argparse

import torch
from torch.nn import Conv2d
from vit_pytorch.cct import CCT
from vit_pytorch.levit import LeViT

from vit_pytorch.t2t import T2TViT

from models import cal_weights_num
from utils.args import add_args


def _t2t(args):
    model = T2TViT(
        dim=512,
        image_size=args.img_size,
        channels=1 if args.dataset_name == 'fmnist' else 3,
        depth=5,
        heads=8,
        mlp_dim=512,
        num_classes=10,
        t2t_layers=((7, 4), (3, 2), (3, 2))
        # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )
    return model


def _levit(args):
    model = LeViT(
        image_size=args.img_size,
        num_classes=10,
        stages=3,  # number of stages
        dim=(256, 384, 512),  # dimensions at each stage
        depth=4,  # transformer of depth 4 at each stage
        heads=(4, 6, 8),  # heads at each stage
        mlp_mult=2,
        dropout=0.1
    )

    if args.dataset_name == 'fmnist' and args.in_channels == 1:
        model.conv_embedding[0] = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    return model


def _cct(args):
    dataset_name = args.dataset_name
    if args.dataset_name == 'fmnist' and args.in_channels == 1:
        input_channel = 1
    else:
        input_channel = 3

    model = CCT(
        n_input_channels=input_channel,
        img_size=(args.img_size, args.img_size),
        embedding_dim=384,
        n_conv_layers=2,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        num_layers=14,
        num_heads=6,
        mlp_radio=3.,
        num_classes=10,
        positional_embedding='learnable',  # ['sine', 'learnable', 'none']
    )
    return model


def _maxvit(args):
    from vit_pytorch.max_vit import MaxViT
    model = MaxViT(
        num_classes=10,
        dim_conv_stem=64,
        dim=96,
        dim_head=32,
        depth=(2, 2, 5, 2),
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1
    )
    return model


def _cait(args):
    from vit_pytorch.cait import CaiT
    model = CaiT(
        image_size=args.img_size,
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=12,  # depth of transformer for patch to patch attention only
        cls_depth=2,  # depth of cross attention of CLS tokens to patch
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05  # randomly dropout 5% of the layers
    )
    return model


def _region_vit(args):
    from vit_pytorch.regionvit import RegionViT
    model = RegionViT(
        dim=(64, 128, 256, 512),
        depth=(2, 2, 8, 2),
        window_size=7,
        num_classes=10,
        tokenize_local_3_conv=False,
        use_peg=False,
    )
    return model


def _mobile(args):
    from vit_pytorch.mobile_vit import MobileViT
    model = MobileViT(
        image_size=(args.img_size, args.img_size),
        dims=[96, 120, 144],
        channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        num_classes=10
    )
    return model


if __name__ == '__main__':
    # model = _levit()
    # model.conv_embedding[0] = Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    # print(model)

    # data_dir = '/data/zxj/dataset/mnist'
    # transform = transforms.Compose([transforms.ToTensor()])
    # server_test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    # dl = data.DataLoader(dataset=server_test_dataset, batch_size=2, shuffle=False, drop_last=False)
    # x = enumerate(dl)
    # x, y = x.__next__()
    # prob = model(y[0])
    # print(prob)
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    args.img_size = 256

    x = torch.randn(1, 1, args.img_size, args.img_size)
    # model = _cct(args)
    # model = _levit(args)
    # model = _region_vit(args)
    model = _mobile(args)
    n_weights = cal_weights_num(model)
    print(str(n_weights) + ' M')

    # print(model(x))
