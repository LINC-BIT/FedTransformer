import logging

import torch
from torch import nn

from utils.data_utils import WarmupCosineSchedule


class CNNTrainer:

    def __init__(self, trainer_id, args, model, train_loader=None, val_loader=None, test_loader=None):
        self.id = trainer_id
        self.args = args

        # data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # model
        self.model = model

        # properties
        self.iteration = 0
        self.lr_trace = []

    def get_lr_trace(self):
        return self.lr_trace

    def train(self):
        model = self.model
        device = self.args.device
        model.to(device)
        model.train()

        # Loss func & optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        if self.args.client_optimizer == "sgd":
            # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=self.args.lr, weight_decay=self.args.wd)
            optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=self.args.lr, weight_decay=self.args.wd)
        elif self.args.client_optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=self.args.lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, amsgrad=True)

        # 07-25-16:23
        # if 'cosine' in self.args.lr_scheduler:
        #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=self.args.warmup_step, t_total=self.args.max_round)

        # Train epochs
        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_loader):
                self.iteration += 1

                x, labels = x.to(device), labels.to(device)
                # labels = labels.long()
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Clip grad
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                batch_loss.append(loss.item())

                # recored lr
                lr = optimizer.param_groups[0]['lr']
                # if self.id == 0:
                    # logging.info(optimizer.param_groups[0].keys())
                    # logging.info(f"-------------------  lr is : {lr}")
                self.lr_trace.append(lr)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def test(self, use_test_data=True):
        model = self.model
        device = self.args.device
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_loader):
                x = x.to(device)
                target = target.to(device)
                # target = target.long()

                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        acc = round(metrics['test_correct']/metrics['test_total'], 3)
        avg_loss = round(metrics['test_loss']/metrics['test_total'], 3)
        logging.info(f"\tRoundTest Loss: {avg_loss}, Test accuracy: {acc}")

        return metrics, acc

    def get_train_sample_num(self):
        if self.train_loader:
            return len(self.train_loader.dataset)
        else:
            logging.info('No training data assigned, return 0')
            return 0

    def update_dataset(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def set_model_params(self, model):
        self.model.load_state_dict(model)

    def get_model_params(self):
        return self.model.state_dict()
