from torchvision import transforms
from torchvision.datasets import FashionMNIST

def data():
    root = "/data/zxj/dataset/fmnist/FMNIST"
    trainset = FashionMNIST(root=root, train=True, download=True)


def get_fmnist_transforms(args):
    MEAN, STD = 0.1307, 0.3081
    if args.in_channels == 3:
        train_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Grayscale(3),
            # transforms.RandomCrop(28, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            # transforms.RandomCrop(28, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))
        ])
        test_transform = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,))
        ])
    return train_transform, test_transform


if __name__ == '__main__':
    pass
