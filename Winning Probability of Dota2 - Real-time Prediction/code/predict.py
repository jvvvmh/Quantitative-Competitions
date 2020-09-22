# coding: utf-8

import argparse
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas
import requests
import scipy.stats as stats
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dota_data import get_lv_from_total_xp, player_death_count, building_states_dict, chatwheel_count, chat_count, \
    purchase_states_dict, items_list, rune_states_dict, DotaData
from model import Model

plt.ioff()


def _extract_lgb_features(m):
    lst = []
    for t in range(len(m['players'][0]['gold_t'])):
        mm = {
            'result': 1 - int(m['radiant_win'])
        }
        for i in range(5):
            assert (m['players'][i]['player_slot'] == i)
            assert (m['players'][i + 5]['player_slot'] == i + 128)
        for i in range(121):
            mm[f'heroes_0_has_{i}'] = 0
            mm[f'heroes_1_has_{i}'] = 0
        for i in range(5):
            mm[f"heroes_0_has_{m['players'][i]['hero_id'] - 1}"] = 1
        for i in range(5, 10):
            mm[f"heroes_1_has_{m['players'][i]['hero_id'] - 1}"] = 1
        for i, v in enumerate(sorted([m['players'][i]['gold_t'][t] for i in range(5)])):
            mm[f"gold_0_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['gold_t'][t] for i in range(5, 10)])):
            mm[f"gold_1_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['xp_t'][t] for i in range(5)])):
            mm[f"xp_0_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['xp_t'][t] for i in range(5, 10)])):
            mm[f"xp_1_{i}"] = v
        for i, v in enumerate(sorted([get_lv_from_total_xp(m['players'][i]['xp_t'][t])
                                      for i in range(5)])):
            mm[f"lv_0_{i}"] = v
        for i, v in enumerate(sorted([get_lv_from_total_xp(m['players'][i]['xp_t'][t])
                                      for i in range(5, 10)])):
            mm[f"lv_1_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['lh_t'][t] for i in range(5)])):
            mm[f"lh_0_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['lh_t'][t] for i in range(5, 10)])):
            mm[f"lh_1_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['dn_t'][t] for i in range(5)])):
            mm[f"dn_0_{i}"] = v
        for i, v in enumerate(sorted([m['players'][i]['dn_t'][t] for i in range(5, 10)])):
            mm[f"dn_1_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['obs_log'] if it['time'] < t * 60)
                                      for i in range(5)])):
            mm[f"obs_0_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['obs_log'] if it['time'] < t * 60)
                                      for i in range(5, 10)])):
            mm[f"obs_1_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['sen_log'] if it['time'] < t * 60)
                                      for i in range(5)])):
            mm[f"sen_0_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['sen_log'] if it['time'] < t * 60)
                                      for i in range(5, 10)])):
            mm[f"sen_1_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['kills_log'] if it['time'] < t * 60)
                                      for i in range(5)])):
            mm[f"kills_0_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for it in m['players'][i]['kills_log'] if it['time'] < t * 60)
                                      for i in range(5, 10)])):
            mm[f"kills_1_{i}"] = v
        for i, v in enumerate(sorted([player_death_count(m, i, t * 60) for i in range(5)])):
            mm[f"deaths_0_{i}"] = v
        for i, v in enumerate(sorted([player_death_count(m, i, t * 60) for i in range(128, 128 + 5)])):
            mm[f"deaths_1_{i}"] = v
        for i, v in enumerate(sorted([sum(tf['players'][i]['damage']
                                          for tf in m['teamfights'] if tf['end'] < t * 60)
                                      for i in range(5)])):
            mm[f"tf_damage_0_{i}"] = v
        for i, v in enumerate(sorted([sum(tf['players'][i]['damage']
                                          for tf in m['teamfights'] if tf['end'] < t * 60)
                                      for i in range(5, 10)])):
            mm[f"tf_damage_1_{i}"] = v
        for k, v in building_states_dict(m, t).items():
            mm[k] = v
        for i, v in enumerate(sorted([chatwheel_count(m, i, t) for i in range(5)])):
            mm[f"chatwheel_0_{i}"] = v
        for i, v in enumerate(sorted([chatwheel_count(m, i, t) for i in range(5, 10)])):
            mm[f"chatwheel_1_{i}"] = v
        for i, v in enumerate(sorted([chat_count(m, i, t) for i in range(5)])):
            mm[f"chat_0_{i}"] = v
        for i, v in enumerate(sorted([chat_count(m, i, t) for i in range(5, 10)])):
            mm[f"chat_1_{i}"] = v
        for i, v in enumerate(sorted([chatwheel_count(m, i, t) + chat_count(m, i, t)
                                      for i in range(5)])):
            mm[f"chat_total_0_{i}"] = v
        for i, v in enumerate(sorted([chatwheel_count(m, i, t) + chat_count(m, i, t)
                                      for i in range(5, 10)])):
            mm[f"chat_total_1_{i}"] = v
        mm['roshan_0'] = sum(1 for o in m['objectives']
                             if o['type'] == 'CHAT_MESSAGE_ROSHAN_KILL'
                             and o['time'] < t * 60 and o['team'] == 2)
        mm['roshan_1'] = sum(1 for o in m['objectives']
                             if o['type'] == 'CHAT_MESSAGE_ROSHAN_KILL'
                             and o['time'] < t * 60 and o['team'] == 3)
        mm['time'] = t
        for k, v in purchase_states_dict(m, items_list, t).items():
            mm[k] = v
        for k, v in rune_states_dict(m, t).items():
            mm[k] = v
        mm['pred_vict_0'] = sum(m['players'][i]['pred_vict'] for i in range(5))
        mm['pred_vict_1'] = sum(m['players'][i]['pred_vict'] for i in range(5, 10))
        for i, v in enumerate(sorted([sum(1 for b in m['players'][i]['buyback_log'] if b['time'] < t * 60)
                                      for i in range(5)])):
            mm[f"buyback_0_{i}"] = v
        for i, v in enumerate(sorted([sum(1 for b in m['players'][i]['buyback_log'] if b['time'] < t * 60)
                                      for i in range(5, 10)])):
            mm[f"buyback_1_{i}"] = v
        mm['region'] = m['region']

        # derivatives (total)
        mm["gold_0_total"] = sum(m['players'][i]['gold_t'][t] for i in range(5))
        mm["gold_1_total"] = sum(m['players'][i]['gold_t'][t] for i in range(5, 10))
        mm["xp_0_total"] = sum(m['players'][i]['xp_t'][t] for i in range(5))
        mm["xp_1_total"] = sum(m['players'][i]['xp_t'][t] for i in range(5, 10))
        mm["lh_0_total"] = sum(m['players'][i]['lh_t'][t] for i in range(5))
        mm["lh_1_total"] = sum(m['players'][i]['lh_t'][t] for i in range(5, 10))
        mm["dn_0_total"] = sum(m['players'][i]['dn_t'][t] for i in range(5))
        mm["dn_1_total"] = sum(m['players'][i]['dn_t'][t] for i in range(5, 10))
        mm["obs_0_total"] = sum(
            sum(1 for it in m['players'][i]['obs_log'] if it['time'] < t * 60) for i in range(5))
        mm["obs_1_total"] = sum(
            sum(1 for it in m['players'][i]['obs_log'] if it['time'] < t * 60) for i in range(5, 10))
        mm["sen_0_total"] = sum(
            sum(1 for it in m['players'][i]['sen_log'] if it['time'] < t * 60) for i in range(5))
        mm["sen_1_total"] = sum(
            sum(1 for it in m['players'][i]['sen_log'] if it['time'] < t * 60) for i in range(5, 10))
        mm["kills_0_total"] = sum(
            sum(1 for it in m['players'][i]['kills_log'] if it['time'] < t * 60) for i in range(5))
        mm["kills_1_total"] = sum(
            sum(1 for it in m['players'][i]['kills_log'] if it['time'] < t * 60) for i in range(5, 10))
        mm["chatwheel_0_total"] = sum(chatwheel_count(m, i, t) for i in range(5))
        mm["chatwheel_1_total"] = sum(chatwheel_count(m, i, t) for i in range(5, 10))
        mm["rune_0_total"] = sum(mm[f"rune_{j}_0_{i}"] for j in range(7) for i in range(5))
        mm["rune_1_total"] = sum(mm[f"rune_{j}_1_{i}"] for j in range(7) for i in range(5))
        mm["buyback_0_total"] = sum(
            sum(1 for b in m['players'][i]['buyback_log'] if b['time'] < t * 60) for i in range(5))
        mm["buyback_1_total"] = sum(
            sum(1 for b in m['players'][i]['buyback_log'] if b['time'] < t * 60) for i in range(5, 10))

        # derivatives (diff)
        mm['pred_vict_diff'] = mm['pred_vict_0'] - mm['pred_vict_1']
        mm['buyback_diff'] = mm['buyback_0_total'] - mm['buyback_1_total']

        # derivatives (ratio)
        for i, v in enumerate(sorted([m['players'][i]['gold_t'][t] for i in range(5)])):
            mm[f"gold_0_ratio_{i}"] = (v + 1) / (mm["gold_0_total"] + mm["gold_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['gold_t'][t] for i in range(5, 10)])):
            mm[f"gold_1_ratio_{i}"] = (v + 1) / (mm["gold_0_total"] + mm["gold_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['xp_t'][t] for i in range(5)])):
            mm[f"xp_0_ratio_{i}"] = (v + 1) / (mm["xp_0_total"] + mm["xp_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['xp_t'][t] for i in range(5, 10)])):
            mm[f"xp_1_ratio_{i}"] = (v + 1) / (mm["xp_0_total"] + mm["xp_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['lh_t'][t] for i in range(5)])):
            mm[f"lh_0_ratio_{i}"] = (v + 1) / (mm["lh_0_total"] + mm["lh_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['lh_t'][t] for i in range(5, 10)])):
            mm[f"lh_1_ratio_{i}"] = (v + 1) / (mm["lh_0_total"] + mm["lh_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['dn_t'][t] for i in range(5)])):
            mm[f"dn_0_ratio_{i}"] = (v + 1) / (mm["dn_0_total"] + mm["dn_1_total"] + 10)
        for i, v in enumerate(sorted([m['players'][i]['dn_t'][t] for i in range(5, 10)])):
            mm[f"dn_1_ratio_{i}"] = (v + 1) / (mm["dn_0_total"] + mm["dn_1_total"] + 10)
        mm["gold_ratio_total"] = (mm["gold_0_total"] + 1) / (mm["gold_0_total"] + mm["gold_1_total"] + 2)
        mm["xp_ratio_total"] = (mm["xp_0_total"] + 1) / (mm["xp_0_total"] + mm["xp_1_total"] + 2)
        mm["lh_ratio_total"] = (mm["lh_0_total"] + 1) / (mm["lh_0_total"] + mm["lh_1_total"] + 2)
        mm["dn_ratio_total"] = (mm["dn_0_total"] + 1) / (mm["dn_0_total"] + mm["dn_1_total"] + 2)
        mm["obs_ratio"] = (mm["obs_0_total"] + 1) / (mm["obs_0_total"] + mm["obs_1_total"] + 2)
        mm["sen_ratio"] = (mm["sen_0_total"] + 1) / (mm["sen_0_total"] + mm["sen_1_total"] + 2)
        mm["kill_ratio"] = (mm["kills_0_total"] + 1) / (mm["kills_0_total"] + mm["kills_1_total"] + 2)
        mm["chatwheel_ratio"] = (mm["chatwheel_0_total"] + 1) / (
                mm["chatwheel_0_total"] + mm["chatwheel_1_total"] + 2)
        mm["rune_ratio"] = (mm["rune_0_total"] + 1) / (mm["rune_0_total"] + mm["rune_1_total"] + 2)

        # derivatives (delta)
        for i, v in enumerate(sorted(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in
                 range(5)])):
            mm[f"gold_delta1_0_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in
                 range(5, 10)])):
            mm[f"gold_delta1_1_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in
                 range(5)])):
            mm[f"gold_delta5_0_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in
                 range(5, 10)])):
            mm[f"gold_delta5_1_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(5)])):
            mm[f"xp_delta1_0_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in
                 range(5, 10)])):
            mm[f"xp_delta1_1_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(5)])):
            mm[f"xp_delta5_0_{i}"] = v
        for i, v in enumerate(sorted(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in
                 range(5, 10)])):
            mm[f"xp_delta5_1_{i}"] = v
        mm["obs_delta6_0"] = sum(
            sum(1 for it in m['players'][i]['obs_log'] if (t - 6) * 60 <= it['time'] < t * 60) for i in
            range(5))
        mm["obs_delta6_1"] = sum(
            sum(1 for it in m['players'][i]['obs_log'] if (t - 6) * 60 <= it['time'] < t * 60) for i in
            range(5, 10))
        mm["sen_delta6_0"] = sum(
            sum(1 for it in m['players'][i]['sen_log'] if (t - 6) * 60 <= it['time'] < t * 60) for i in
            range(5))
        mm["sen_delta6_1"] = sum(
            sum(1 for it in m['players'][i]['sen_log'] if (t - 6) * 60 <= it['time'] < t * 60) for i in
            range(5, 10))

        # derivatives (delta_total)
        mm["gold_delta1_total_0"] = sum(
            m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in range(5))
        mm["gold_delta1_total_1"] = sum(
            m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in range(5, 10))
        mm["gold_delta5_total_0"] = sum(
            m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in range(5))
        mm["gold_delta5_total_1"] = sum(
            m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in range(5, 10))
        mm["xp_delta1_total_0"] = sum(
            m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(5))
        mm["xp_delta1_total_1"] = sum(
            m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(5, 10))
        mm["xp_delta5_total_0"] = sum(
            m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(5))
        mm["xp_delta5_total_1"] = sum(
            m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(5, 10))

        # derivatives (delta_ratio)
        mm["gold_delta1_ratio_total"] = (mm["gold_delta1_total_0"] + 1) / (
                mm["gold_delta1_total_0"] + mm["gold_delta1_total_1"] + 2)
        mm["gold_delta5_ratio_total"] = (mm["gold_delta5_total_0"] + 1) / (
                mm["gold_delta5_total_0"] + mm["gold_delta5_total_1"] + 2)
        mm["xp_delta1_ratio_total"] = (mm["xp_delta1_total_0"] + 1) / (
                mm["xp_delta1_total_0"] + mm["xp_delta1_total_1"] + 2)
        mm["xp_delta5_ratio_total"] = (mm["xp_delta5_total_0"] + 1) / (
                mm["xp_delta5_total_0"] + mm["xp_delta5_total_1"] + 2)

        # derivatives (mean)
        for i, v in enumerate(
                sorted([0 if t == 0 else m['players'][i]['gold_t'][t] / t for i in range(5)])):
            mm[f"gpm_0_{i}"] = v
        for i, v in enumerate(
                sorted([0 if t == 0 else m['players'][i]['gold_t'][t] / t for i in range(5, 10)])):
            mm[f"gpm_1_{i}"] = v
        for i, v in enumerate(sorted([0 if t == 0 else m['players'][i]['xp_t'][t] / t for i in range(5)])):
            mm[f"xpm_0_{i}"] = v
        for i, v in enumerate(
                sorted([0 if t == 0 else m['players'][i]['xp_t'][t] / t for i in range(5, 10)])):
            mm[f"xpm_1_{i}"] = v

        # derivatives (variation)
        mm["gold_variation_0"] = stats.variation([m['players'][i]['gold_t'][t] + 1.0 for i in range(5)])
        mm["gold_variation_1"] = stats.variation([m['players'][i]['gold_t'][t] + 1.0 for i in range(5, 10)])
        mm["xp_variation_0"] = stats.variation([m['players'][i]['xp_t'][t] + 1.0 for i in range(5)])
        mm["xp_variation_1"] = stats.variation([m['players'][i]['xp_t'][t] + 1.0 for i in range(5, 10)])

        lst.append(mm)
    return pandas.DataFrame(lst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("match_id", type=int)
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

    root_url = 'https://api.opendota.com/api'
    r = requests.post(f"{root_url}/request/{params.match_id}")
    if not r.ok:
        print(f'Requesting {root_url}/request/{params.match_id} failed, aborting.')
        exit(-1)
    time.sleep(3)
    r = requests.get(f"{root_url}/matches/{params.match_id}")
    retry = 0
    while not r.ok or len(r.content) < 30000:
        if retry > 5:
            print(f'Requesting {root_url}/matches/{params.match_id} failed, aborting.')
            exit(-1)
        print(f'Requesting {root_url}/matches/{params.match_id} failed, retrying...')
        retry += 1
        time.sleep(3)
        r = requests.get(f"{root_url}/matches/{params.match_id}")

    m = r.json()
    m = DotaData.transform_test(m)
    si, hero, sg, x_len, e, e_t, e_mask, e_len, y, z, m_id = DotaData.collate_fn_with_id([m])

    model = Model(params.d_si, params.d_sg, params.d, params.d_hid, params.n_h, params.n_layers)
    model.load_state_dict(torch.load(params.path_model_1, map_location=lambda storage, _: storage)['state_dict'])

    with torch.no_grad():
        prob_1, _, _ = model(si, hero, sg, e, e_t, e_mask, e_len)
    prob_1 = torch.sigmoid(prob_1).view(-1)

    m = r.json()
    m = _extract_lgb_features(m)
    m.region = m.region.astype('category')

    lgb_model = lgb.Booster(model_file=params.path_model_0)
    prob_0 = torch.FloatTensor(lgb_model.predict(m.drop('result', axis=1))).view(-1)

    prob_2 = 1.0 - (prob_0 + prob_1) * 0.5

    fig = plt.figure(figsize=(10, 5))
    plt.xlim([0 - 2, prob_2.size(0) - 2])
    plt.ylim([0.0, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.plot(np.arange(prob_2.size(0)) - 2, np.array(prob_2), color='r')
    fig.savefig(f'{params.match_id}.png', transparent=True, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
