import logging

from utils.data_utils import record_net_data_stats, NET_DATA_STATE


class Client:

    def __init__(self, client_idx, client_dl_dict, model_trainer):
        self.client_idx = client_idx

        # dict of data_loader of clients
        self.client_dl_dict = client_dl_dict

        # trainer for training
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx):
        self.client_idx = client_idx
        self.model_trainer.update_dataset(
            train_loader=self.client_dl_dict[client_idx][0],
            test_loader=self.client_dl_dict[client_idx][1]
        )

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train()
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset=True):
        metrics = self.model_trainer.test(b_use_test_dataset)
        return metrics

    def get_sample_number(self):
        sample_num = self.model_trainer.get_train_sample_num()
        logging.debug(f'Client idx: {self.client_idx} has trained {sample_num} samples.')
        logging.info(f'Data statistics of client_idx-{self.client_idx}: {NET_DATA_STATE[self.client_idx]}')
        return sample_num

    def get_lr_trace(self):
        return self.model_trainer.get_lr_trace()
