import copy
import logging
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from client import Client

#  ============
from models.ctrainer import CNNTrainer
from models.vtrainer import ViTTrainer
from utils.args import show_setting, save_setting
from utils.time import current_datetime_string, record_time


def client_sampling(round_idx, client_num_in_total, client_num_per_round):
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    # logging.info("client_indexes = %s" % str(client_indexes))
    return client_indexes


def _aggregate(w_locals):
    training_num = 0
    for idx in range(len(w_locals)):
        (sample_num, averaged_params) = w_locals[idx]
        training_num += sample_num

    (sample_num, averaged_params) = w_locals[0]
    for k in averaged_params.keys():
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    return averaged_params


TRAINER_CLS = {
    "cnn": CNNTrainer,
    "vit": ViTTrainer
}


class Server:
    def __init__(self,
                 client_dl_dict,
                 server_test_dl,
                 model,
                 args):
        """
        Manage the whole federated learning progress.
        Args:
            client_dl_dict(dict): loaders[0], [1]  -->   client_train_dl_dict, client_test_dl_dict
            server_test_dl(DataLoader):
            model: cnn model or vit model
            args: parsed configs
        """
        self.setting = show_setting(args)
        self.writer = None  # traces writer

        self.working_clients = []
        self.client_dl_dict = client_dl_dict
        self.server_test_loader = server_test_dl
        self.model = model
        self.args = args
        self.best_acc = 0

        ModelTrainer = TRAINER_CLS[args.model_type]
        self.server_trainer = ModelTrainer(trainer_id=-1, args=self.args, model=self.model, test_loader=self.server_test_loader)
        self._setup_clients(ModelTrainer)

    def _setup_clients(self, ModelTrainer):
        logging.info("============ setup_clients (START) ============")
        for client_idx in range(self.args.client_num_per_round):
            client_trainer = ModelTrainer(trainer_id=client_idx, args=self.args, model=self.model)
            c = Client(client_idx, self.client_dl_dict, client_trainer)
            self.working_clients.append(c)
        logging.info("============ setup_clients (END) ============")

    @record_time
    def train(self):
        # Get global model
        w_global = self._get_model_params()

        # Start federated training
        for round_idx in range(self.args.max_round):
            logging.info("################  Communication round : {}".format(round_idx))

            # Select part of clients for training
            client_indexes = client_sampling(round_idx,  self.args.client_num_in_total, self.args.client_num_per_round)
            logging.info("Current round's client_indexes = " + str(client_indexes))

            # Update client data and train on the data
            w_locals = []
            for idx, client in enumerate(self.working_clients):
                # Update dataset
                client_data_idx = client_indexes[idx]
                client.update_local_dataset(client_data_idx)

                # Train on the new dataset
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # Average client models and update the global model
            w_global = _aggregate(w_locals)
            self._set_model_params(w_global)

            # Test or evaluate the global model
            # at last round
            if round_idx == self.args.max_round - 1:
                self.global_test(round_idx)

            # at eval round
            elif round_idx % self.args.test_freq == 0:
                self.global_test(round_idx)

        self.writer.close()

    def global_test(self, round_idx, show_client_lr=False):
        if not self.writer:
            self.output_dir = f'./experiments/logs/{self.args.dataset_name}-{self.args.model}-lr{self.args.lr}' \
                              f'-wd{self.args.wd}-part-{self.args.partition}-img_size{self.args.img_size}' \
                              f'-total_c{self.args.client_num_in_total}-per_c{self.args.client_num_per_round}' \
                         f'-train_bs{self.args.train_bs}-maxr{self.args.max_round}-{current_datetime_string()}'
            os.makedirs(self.output_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "report"))
            save_setting(self.setting, self.output_dir)

        # record global test metric
        logging.info(f'\tRound {round_idx} metrics: ')
        _, acc = self.server_trainer.test()
        self.writer.add_scalar("global/test_accuracy", scalar_value=acc, global_step=round_idx)
        # record best acc
        if acc > self.best_acc:
            self.best_acc = acc

        if show_client_lr:
            client = self.working_clients[0]
            trace = client.get_lr_trace()
            for i, lr in enumerate(trace):
                self.writer.add_scalar("client0/lr", scalar_value=lr, global_step=i)

    def _get_model_params(self):
        return self.server_trainer.get_model_params()

    def _set_model_params(self, w_global):
        self.server_trainer.set_model_params(w_global)

