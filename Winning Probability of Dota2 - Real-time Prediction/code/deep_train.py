# coding: utf-8

import argparse
import logging
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dota_data import DotaData
from model import Model

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')


class Trainer:
    def __init__(self, params):
        self.data_loader_train = DataLoader(
            DotaData(params.path_data, 'train', transform=DotaData.transform_train),
            batch_size=params.bs, num_workers=12, shuffle=True, pin_memory=True, collate_fn=DotaData.collate_fn)
        self.data_loader_valid = DataLoader(
            DotaData(params.path_data, 'valid', transform=DotaData.transform_valid),
            batch_size=params.bs, num_workers=12, shuffle=False, pin_memory=True, collate_fn=DotaData.collate_fn)
        self.model = Model(params.d_si, params.d_sg, params.d, params.d_hid, params.n_h, params.n_layers,
                           drop_prob=params.drop_prob).to(GPU)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr, betas=(params.lr_beta1, params.lr_beta2),
                                    weight_decay=params.wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=0, factor=params.lr_decay,
                                                              verbose=True)

    def epoch_train(self):
        self.model.train()
        tot, cnt = 0.0, 0
        for si, hero, sg, x_len, e, e_t, e_mask, e_len, y, _ in tqdm(self.data_loader_train):
            si, hero, sg, x_len, e, e_t, e_mask, e_len, y = si.to(GPU), hero.to(GPU), sg.to(GPU), x_len.to(GPU), e.to(
                GPU), e_t.to(GPU), e_mask.to(GPU), e_len.to(GPU), y.to(GPU)
            self.optimizer.zero_grad()
            pred, _, _ = self.model(si, hero, sg, e, e_t, e_mask, e_len)
            loss = Model.loss_fn(pred, x_len, y.to(torch.float))
            loss.backward()
            self.optimizer.step()
            logging.info(loss.item())
            tot += loss.item()
            cnt += 1
        return tot / cnt

    def epoch_valid(self):
        self.model.eval()
        correct, total, total_loss, cnt = 0, 0, 0, 0
        with torch.no_grad():
            for si, hero, sg, x_len, e, e_t, e_mask, e_len, y, dur in self.data_loader_valid:
                si, hero, sg, x_len, e, e_t, e_mask, e_len, y, dur = si.to(GPU), hero.to(GPU), sg.to(GPU), x_len.to(
                    GPU), e.to(GPU), e_t.to(GPU), e_mask.to(GPU), e_len.to(GPU), y.to(GPU), dur.to(GPU)
                pred, _, _ = self.model(si, hero, sg, e, e_t, e_mask, e_len)
                total_loss += Model.loss_fn(pred, x_len, y.to(torch.float))
                cnt += 1
                c, t = Model.accuracy(pred, y, dur)
                correct += c
                total += t
        return total_loss / cnt, correct / total


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", type=str)
    parser.add_argument("--out", type=str, default=timestamp)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--d_si", type=int, default=88)
    parser.add_argument("--d_sg", type=int, default=103)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--d_hid", type=int, default=320)
    parser.add_argument("--n_h", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--drop_prob", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_beta1", type=float, default=0.9)
    parser.add_argument("--lr_beta2", type=float, default=0.999)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--wd", type=float, default=3e-6)
    parser.add_argument("--n_epochs", type=int, default=4)
    params = parser.parse_args()
    os.mkdir(params.out)
    print(params)

    logging.basicConfig(filename=os.path.join(params.out, 'log.txt'), level=logging.DEBUG)

    trainer = Trainer(params)
    best_acc = 0.0
    for epoch in range(params.n_epochs):
        train_loss = trainer.epoch_train()
        valid_loss, valid_acc = trainer.epoch_valid()
        trainer.scheduler.step(valid_loss)
        logging.warning(f'train_loss = {train_loss} valid_loss = {valid_loss},  acc = {valid_acc}')
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                'loss': valid_loss,
                'acc': valid_acc,
                'state_dict': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict()
            }, os.path.join(params.out, f"epoch{epoch}.pt"))

    print(params)


if __name__ == '__main__':
    main()
