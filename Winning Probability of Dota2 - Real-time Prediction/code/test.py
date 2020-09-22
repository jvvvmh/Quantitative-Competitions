# coding: utf-8

import argparse

import lightgbm as lgb
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dota_data import DotaData, load_as_data_frame
from model import Model

CPU = torch.device('cpu')


def _pred_to_10pct(pred, dur):  # pred: Float[bs, max_T], y: Long[bs], dur: Float[bs]
    pct = torch.FloatTensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to(dur.device)
    pos = (torch.ger(dur, pct) / 60.0).to(torch.long)  # pos: Long[bs, 9]
    pred = torch.gather(pred, dim=1, index=pos)  # pred: Float[bs, 9]
    return torch.sigmoid(pred)


class Tester:
    def __init__(self, params):
        self.data_loader_test = DataLoader(
            DotaData(params.path_data, 'test', transform=DotaData.transform_test),
            batch_size=params.bs, num_workers=12, shuffle=False, collate_fn=DotaData.collate_fn_with_id)
        self.model = Model(params.d_si, params.d_sg, params.d, params.d_hid, params.n_h, params.n_layers,
                           drop_prob=params.drop_prob)
        state_dict = torch.load(params.path_model_1, map_location=lambda storage, _: storage)['state_dict']
        self.model.load_state_dict(state_dict)
        print(f"Number of parameters: {sum(pa.numel() for pa in self.model.parameters() if pa.requires_grad)}")

    def predict(self):
        self.model.eval()
        lst, l2, li, ly = [], [], [], []
        with torch.no_grad():
            for si, hero, sg, x_len, e, e_t, e_mask, e_len, y, dur, m_id in tqdm(self.data_loader_test):
                pred, _, _ = self.model(si, hero, sg, e, e_t, e_mask, e_len)
                lst.append(_pred_to_10pct(pred, dur))
                l2.append(pred[:, 20])
                li.append(m_id)
                ly.append(y)
        return torch.cat(lst), torch.cat(l2), torch.cat(li), torch.cat(ly)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", type=str)
    parser.add_argument("path_model_0", type=str)
    parser.add_argument("path_model_1", type=str)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--d_si", type=int, default=88)
    parser.add_argument("--d_sg", type=int, default=103)
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--d_hid", type=int, default=320)
    parser.add_argument("--n_h", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--drop_prob", type=float, default=0)
    params = parser.parse_args()

    df, id_0 = load_as_data_frame(params.path_data, ['test'])
    df_test = df['test']
    lgb_model = lgb.Booster(model_file=params.path_model_0)
    prob_0 = torch.FloatTensor(lgb_model.predict(df_test.drop('result', axis=1))).view(-1, 9)
    y_0 = torch.LongTensor(df_test.result).view(-1, 9)[:, 0]
    pred_0 = (prob_0 >= 0.5).to(torch.long)
    cor = (pred_0 == y_0.unsqueeze(-1).expand_as(pred_0)).to(torch.float)
    cor = cor.mean(dim=0)
    print(f"LightGBM model acc = {cor.tolist()}")
    print(f"LightGBM model total acc = {cor.mean()}")

    tester = Tester(params)
    prob_1, prob_20_1, id_1, y_1 = tester.predict()
    pred_1 = (prob_1 >= 0.5).to(torch.long)
    pred_20_1 = (prob_20_1 >= 0.5).to(torch.long)
    cor = (pred_1 == y_1.unsqueeze(-1).expand_as(pred_1)).to(torch.float)
    cor = cor.mean(dim=0)
    print(f"Deep model acc = {cor.tolist()}")
    print(f"Deep model total acc = {cor.mean()}")
    cor = (pred_20_1 == y_1).to(torch.float).mean()
    print(f"Deep model 20min acc = {cor}")

    id_0 = np.argsort(np.array(id_0))
    id_1 = np.argsort(np.array(id_1))
    y_0 = y_0[id_0]
    y_1 = y_1[id_1]
    prob_0 = prob_0[id_0]
    prob_1 = prob_1[id_1]
    prob_2 = (prob_0 + prob_1) * 0.5
    if not (y_0 == y_1).all():
        raise Exception("error")
    y_2 = y_0
    pred_2 = (prob_2 >= 0.5).to(torch.long)
    cor = (pred_2 == y_2.unsqueeze(-1).expand_as(pred_2)).to(torch.float)
    cor = cor.mean(dim=0)
    print(f"Ensemble acc = {cor.tolist()}")
    print(f"Ensemble total acc = {cor.mean()}")


if __name__ == '__main__':
    main()
