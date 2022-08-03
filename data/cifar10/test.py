import numpy as np
import torchvision

from data.partition import CIFAR10Partitioner


def cifar10_partitioner(is_train, client_num=100, n_shards=200):
    data_dir = 'E:\\PyProjects\\datasets\\cifar10'
    balance = None
    partition = 'shards'
    alpha = 0.1
    seed = 2021

    dataset = torchvision.datasets.CIFAR10(root=data_dir, train=is_train, download=True)

    shards_part = CIFAR10Partitioner(dataset.targets,
                                           client_num,
                                           balance=balance,
                                           partition=partition,
                                           num_shards=n_shards,
                                           dir_alpha=alpha,
                                           seed=seed)
    return dataset, shards_part


if __name__ == '__main__':
    from data.functional import partition_report
    import pandas as pd
    import matplotlib.pyplot as plt

    dataset, shards_part = cifar10_partitioner(False)
    # generate partition report
    csv_file = "./partition-reports/cifar10_shards_200_100clients.csv"
    n_class = 10
    partition_report(dataset.targets, shards_part.client_dict, class_num=n_class, verbose=False, file=csv_file)

    shards_part_df = pd.read_csv(csv_file, header=1)
    shards_part_df = shards_part_df.set_index('client')
    col_names = [f"class{i}" for i in range(n_class)]
    for col in col_names:
        shards_part_df[col] = (shards_part_df[col] * shards_part_df['Amount']).astype(int)

    # select first 10 clients for bar plot
    shards_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('sample num')
    plt.savefig(f"./imgs/cifar10_shards_200_100clients.png", dpi=400, bbox_inches='tight')