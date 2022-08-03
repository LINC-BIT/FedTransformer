import argparse
import logging
import math

import torch
import torchvision
import torchvision.datasets.utils
from torch.utils import data
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

from data.cifar10.cifar_loader import CIFAR10_Truncated, get_data_transforms_cifar10
from data.fmnist.fmnist_loader import get_fmnist_transforms
from data.partition import CIFAR10Partitioner, MNISTPartitioner, FMNISTPartitioner
from data.sampler import SubsetSampler
from models import cal_weights_num
from models.cnn import simple_cnn_cifar10, _resnet50, _regnety_4g, _efficient_b4

######################################################################
# num_workers per client
# homo
from models.vit import _cct, _levit, _t2t

NUM_WORKERS_RANK_DIC = [16]
# NUM_WORKERS_RANK_DIC = [4 for _ in range(10)]
# NUM_WORKERS_RANK_DIC.insert(0, 6)
######################################################################

MODELS = {
    'levit': _levit,
    'simplecnn': simple_cnn_cifar10,
    'regnety_4g': _regnety_4g,
    'efficient_b4': _efficient_b4,
    'resnet50': _resnet50,
    'cct': _cct,
    't2t': _t2t
}
######################################################################
NET_DATA_STATE = {}


def load_data(args,
              test_bs=512,
              balance=None,
              partition="iid",
              alpha=None,
              seed=2022):
    """Make data loaders for clients or server

    Args:
        args(NameSpace):
        test_bs: test batch size
        balance:  None, False or True; None -> diri & iid;
        partition: iid diri shards
        alpha:
        seed:
    """
    dataset_name, data_dir = args.dataset_name, args.data_dir
    client_num = args.client_num_in_total
    logging.info("load_data. dataset_name = {}\n data_dir is {} \t "
                 "train_batch_size is {}".format(dataset_name, data_dir, args.train_bs))

    client_dl_dict = {}
    if dataset_name == 'cifar10':
        # transform_train, transform_test = get_data_transforms_cifar10_v2(args.img_size)
        transform_train, transform_test = get_data_transforms_cifar10()

        # Server data loader
        server_test_dataset = CIFAR10_Truncated(data_dir, dataidxs=None, train=False, transform=transform_test,
                                                download=True)
        server_test_dl = data.DataLoader(dataset=server_test_dataset, batch_size=test_bs, shuffle=False,
                                         drop_last=False,
                                         num_workers=NUM_WORKERS_RANK_DIC[0])

        # Client data loader
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

        # partitioned dataset    # client_num_total = 100
        n_shards = 20 if partition == 'split-2' else 40
        if partition == 'iid':
            balance = True
        else:
            partition = 'shards'
        train_client_idxes = CIFAR10Partitioner(
            train_dataset.targets,
            client_num,
            balance=balance,
            partition=partition,
            num_shards=n_shards,
            dir_alpha=alpha,
            seed=seed
        ).client_dict

        test_client_idxes = CIFAR10Partitioner(
            test_dataset.targets,
            client_num,
            balance=balance,
            partition=partition,
            num_shards=n_shards,
            dir_alpha=alpha,
            seed=seed
        ).client_dict

        for client_idx in train_client_idxes.keys():
            # n_workers = NUM_WORKERS_RANK_DIC[client_idx+1]
            n_workers = NUM_WORKERS_RANK_DIC[0]
            train_data_idxes, test_data_idxes = train_client_idxes[client_idx], test_client_idxes[client_idx]
            NET_DATA_STATE[client_idx] = record_net_data_stats(client_idx, train_dataset.targets, train_data_idxes)

            # get truncated dataset
            train_ds = CIFAR10_Truncated(data_dir, dataidxs=train_data_idxes, train=True, transform=transform_train,
                                         download=True)
            test_ds = CIFAR10_Truncated(data_dir, dataidxs=test_data_idxes, train=False, transform=transform_test,
                                        download=True)

            # make data loader
            train_data_shuffle = True
            train_dl = data.DataLoader(
                dataset=train_ds,
                batch_size=args.train_bs,
                shuffle=train_data_shuffle,
                drop_last=False,
                # speed up
                num_workers=n_workers,
                pin_memory=True,
                # prefetch_factor=True
            )
            test_dl = data.DataLoader(
                dataset=test_ds,
                batch_size=test_bs,
                shuffle=False,
                drop_last=False,
                # speed up
                num_workers=n_workers,
                pin_memory=True,
                # prefetch_factor=True
            )
            client_dl_dict[client_idx] = (train_dl, test_dl)

    elif dataset_name == 'fmnist':
        train_transform, test_transform = get_fmnist_transforms(args)
        # Server
        server_test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False,
                                                                transform=test_transform)
        server_test_dl = data.DataLoader(dataset=server_test_dataset, batch_size=test_bs, shuffle=False,
                                         drop_last=False,
                                         num_workers=NUM_WORKERS_RANK_DIC[0])
        # Client
        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=train_transform)

        if partition == 'split-1':
            class_num_per_client = 4
            partition = 'noniid-#label'
        elif partition == 'split-2':
            class_num_per_client = 2
            partition = 'noniid-#label'
        else:
            class_num_per_client = None
            partition = 'iid'

        train_client_idxes = FMNISTPartitioner(train_dataset.targets,
                                               num_clients=args.client_num_in_total,
                                               partition=partition,
                                               major_classes_num=class_num_per_client,
                                               seed=seed).client_dict
        for client_idx in train_client_idxes.keys():
            n_workers = NUM_WORKERS_RANK_DIC[0]
            train_data_idxes = train_client_idxes[client_idx]
            NET_DATA_STATE[client_idx] = record_net_data_stats(client_idx, train_dataset.targets, train_data_idxes)

            # get truncated dataset
            train_dl = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=SubsetSampler(indices=train_data_idxes, shuffle=True),
                                                   batch_size=args.train_bs,
                                                   pin_memory=True,
                                                   num_workers=n_workers,
                                                   drop_last=False)
            test_dl = None

            client_dl_dict[client_idx] = (train_dl, test_dl)
    else:
        raise NotImplementedError("Unsupported dataset")

    return client_dl_dict, server_test_dl


def record_net_data_stats(client_idx, y_train, dataidx):
    """

    Args:
        y_train: target of the whole dataset
        net_dataidx_map: {client_idx: [data_idx]}
    """
    y_train = np.asarray(y_train)
    dataidx = np.asarray(dataidx)
    unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    return tmp


######################################################################
###                     Utils for training                         ###
######################################################################

def create_model(args):
    model = MODELS[args.model](args)
    logging.info(f'Use {model.__class__}')
    model.to(args.device)

    # get the number of model parameter
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.model_quantity = cal_weights_num(model)
    return model


# lr scheduler
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


if __name__ == '__main__':
    from utils.args import add_args

    parser = add_args(argparse.ArgumentParser(description='Data util test'))
    args = parser.parse_args()
    args.partition = 'dirichlet'
    args.client_num_in_total = 10
    args.diri_alpha = 0.2
    print(args)

    client_dl_dict, server_test_dl = load_data(args,
                                               partition=args.partition,
                                               balance=None if args.partition == 'shards' else args.balance,
                                               n_shards=None if not args.num_shards else args.num_shards,
                                               alpha=args.diri_alpha)
    print(data)
