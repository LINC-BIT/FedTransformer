import argparse
import logging
import os
import random

import numpy as np
import setproctitle

import torch

from server import Server
from utils.args import add_args
from utils.data_utils import load_data, create_model


def main():
    # Basic setting
    str_process_name = "FedAvg"
    setproctitle.setproctitle(str_process_name)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = device
    # logger.info(args)

    # Fix the random seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # Load data
    client_dl_dict, server_test_dl = load_data(args, partition=args.partition)

    # Create model
    model = create_model(args)

    # Start federated learning
    server = Server(client_dl_dict, server_test_dl, model, args)
    sys_cost = server.train()

    # Record sys cost metrics
    sys_cost_path = os.path.join(server.output_dir, 'sys_metrics')
    with open(sys_cost_path, 'wt') as f:
        final_msg = 'Finish training: \n'
        final_msg += f'Best global test_acc: {server.best_acc} \n'
        final_msg += f'Training time cost: {str(sys_cost[1])}s \n'
        f.write(final_msg)


if __name__ == '__main__':
    main()
