import json
import os
from multiprocessing import Pool

import numpy as np
import pandas
import scipy.stats as stats
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ['DotaData', 'load_as_data_frame']


def _check_path_match(s):
    """ Random hash to split data set to train, valid, and test. """
    if not s.endswith('.json'):
        return 'none'
    u = int(s[:-5]) % 1000000021
    u = (u * 479342492 + 277101274) % 1000000021
    v = int(s[:-5]) % 1000000033
    v = (v * 27448077 + 702331637) % 1000000033
    w = ((u + v) * 897630631 + 28665357) % 1000000087
    w = w / (1000000087.0 - 1.0)
    if w < 0.05:
        return 'test'
    elif 0.05 <= w < 0.15:
        return 'valid'
    else:
        return 'train'


lv_xp_req = [
    0,
    230,
    600,
    1080,
    1680,
    2300,
    2940,
    3600,
    4280,
    5080,
    5900,
    6740,
    7640,
    8865,
    10115,
    11390,
    12690,
    14015,
    15415,
    16905,
    18405,
    20155,
    22155,
    24405,
    26905,
]

hero2id = {
    'npc_dota_hero_antimage': 1,
    'npc_dota_hero_axe': 2,
    'npc_dota_hero_bane': 3,
    'npc_dota_hero_bloodseeker': 4,
    'npc_dota_hero_crystal_maiden': 5,
    'npc_dota_hero_drow_ranger': 6,
    'npc_dota_hero_earthshaker': 7,
    'npc_dota_hero_juggernaut': 8,
    'npc_dota_hero_mirana': 9,
    'npc_dota_hero_nevermore': 11,
    'npc_dota_hero_morphling': 10,
    'npc_dota_hero_phantom_lancer': 12,
    'npc_dota_hero_puck': 13,
    'npc_dota_hero_pudge': 14,
    'npc_dota_hero_razor': 15,
    'npc_dota_hero_sand_king': 16,
    'npc_dota_hero_storm_spirit': 17,
    'npc_dota_hero_sven': 18,
    'npc_dota_hero_tiny': 19,
    'npc_dota_hero_vengefulspirit': 20,
    'npc_dota_hero_windrunner': 21,
    'npc_dota_hero_zuus': 22,
    'npc_dota_hero_kunkka': 23,
    'npc_dota_hero_lina': 25,
    'npc_dota_hero_lich': 31,
    'npc_dota_hero_lion': 26,
    'npc_dota_hero_shadow_shaman': 27,
    'npc_dota_hero_slardar': 28,
    'npc_dota_hero_tidehunter': 29,
    'npc_dota_hero_witch_doctor': 30,
    'npc_dota_hero_riki': 32,
    'npc_dota_hero_enigma': 33,
    'npc_dota_hero_tinker': 34,
    'npc_dota_hero_sniper': 35,
    'npc_dota_hero_necrolyte': 36,
    'npc_dota_hero_warlock': 37,
    'npc_dota_hero_beastmaster': 38,
    'npc_dota_hero_queenofpain': 39,
    'npc_dota_hero_venomancer': 40,
    'npc_dota_hero_faceless_void': 41,
    'npc_dota_hero_skeleton_king': 42,
    'npc_dota_hero_death_prophet': 43,
    'npc_dota_hero_phantom_assassin': 44,
    'npc_dota_hero_pugna': 45,
    'npc_dota_hero_templar_assassin': 46,
    'npc_dota_hero_viper': 47,
    'npc_dota_hero_luna': 48,
    'npc_dota_hero_dragon_knight': 49,
    'npc_dota_hero_dazzle': 50,
    'npc_dota_hero_rattletrap': 51,
    'npc_dota_hero_leshrac': 52,
    'npc_dota_hero_furion': 53,
    'npc_dota_hero_life_stealer': 54,
    'npc_dota_hero_dark_seer': 55,
    'npc_dota_hero_clinkz': 56,
    'npc_dota_hero_omniknight': 57,
    'npc_dota_hero_enchantress': 58,
    'npc_dota_hero_huskar': 59,
    'npc_dota_hero_night_stalker': 60,
    'npc_dota_hero_broodmother': 61,
    'npc_dota_hero_bounty_hunter': 62,
    'npc_dota_hero_weaver': 63,
    'npc_dota_hero_jakiro': 64,
    'npc_dota_hero_batrider': 65,
    'npc_dota_hero_chen': 66,
    'npc_dota_hero_spectre': 67,
    'npc_dota_hero_doom_bringer': 69,
    'npc_dota_hero_ancient_apparition': 68,
    'npc_dota_hero_ursa': 70,
    'npc_dota_hero_spirit_breaker': 71,
    'npc_dota_hero_gyrocopter': 72,
    'npc_dota_hero_alchemist': 73,
    'npc_dota_hero_invoker': 74,
    'npc_dota_hero_silencer': 75,
    'npc_dota_hero_obsidian_destroyer': 76,
    'npc_dota_hero_lycan': 77,
    'npc_dota_hero_brewmaster': 78,
    'npc_dota_hero_shadow_demon': 79,
    'npc_dota_hero_lone_druid': 80,
    'npc_dota_hero_chaos_knight': 81,
    'npc_dota_hero_meepo': 82,
    'npc_dota_hero_treant': 83,
    'npc_dota_hero_ogre_magi': 84,
    'npc_dota_hero_undying': 85,
    'npc_dota_hero_rubick': 86,
    'npc_dota_hero_disruptor': 87,
    'npc_dota_hero_nyx_assassin': 88,
    'npc_dota_hero_naga_siren': 89,
    'npc_dota_hero_keeper_of_the_light': 90,
    'npc_dota_hero_wisp': 91,
    'npc_dota_hero_visage': 92,
    'npc_dota_hero_slark': 93,
    'npc_dota_hero_medusa': 94,
    'npc_dota_hero_troll_warlord': 95,
    'npc_dota_hero_centaur': 96,
    'npc_dota_hero_magnataur': 97,
    'npc_dota_hero_shredder': 98,
    'npc_dota_hero_bristleback': 99,
    'npc_dota_hero_tusk': 100,
    'npc_dota_hero_skywrath_mage': 101,
    'npc_dota_hero_abaddon': 102,
    'npc_dota_hero_elder_titan': 103,
    'npc_dota_hero_legion_commander': 104,
    'npc_dota_hero_ember_spirit': 106,
    'npc_dota_hero_earth_spirit': 107,
    'npc_dota_hero_terrorblade': 109,
    'npc_dota_hero_phoenix': 110,
    'npc_dota_hero_oracle': 111,
    'npc_dota_hero_techies': 105,
    'npc_dota_hero_target_dummy': 127,
    'npc_dota_hero_winter_wyvern': 112,
    'npc_dota_hero_arc_warden': 113,
    'npc_dota_hero_abyssal_underlord': 108,
    'npc_dota_hero_monkey_king': 114,
    'npc_dota_hero_pangolier': 120,
    'npc_dota_hero_dark_willow': 119,
    'npc_dota_hero_grimstroke': 121
}

items_list = [
    'aegis',
    'courier',
    'boots_of_elves',
    'belt_of_strength',
    'blade_of_alacrity',
    'blades_of_attack',
    'blight_stone',
    'blink',
    'boots',
    'bottle',
    'broadsword',
    'chainmail',
    'cheese',
    'circlet',
    'clarity',
    'claymore',
    'cloak',
    'demon_edge',
    'dust',
    'eagle',
    'enchanted_mango',
    'energy_booster',
    'faerie_fire',
    'flying_courier',
    'gauntlets',
    'gem',
    'ghost',
    'gloves',
    'flask',
    'helm_of_iron_will',
    'hyperstone',
    'infused_raindrop',
    'branches',
    'javelin',
    'magic_stick',
    'mantle',
    'mithril_hammer',
    'lifesteal',
    'mystic_staff',
    'ward_observer',
    'ogre_axe',
    'orb_of_venom',
    'platemail',
    'point_booster',
    'quarterstaff',
    'quelling_blade',
    'reaver',
    'refresher_shard',
    'ring_of_health',
    'ring_of_protection',
    'ring_of_regen',
    'robe',
    'relic',
    'sobi_mask',
    'ward_sentry',
    'shadow_amulet',
    'slippers',
    'smoke_of_deceit',
    'staff_of_wizardry',
    'stout_shield',
    'talisman_of_evasion',
    'tango',
    'tango_single',
    'tome_of_knowledge',
    'tpscroll',
    'ultimate_orb',
    'vitality_booster',
    'void_stone',
    'wind_lace',
]


def get_lv_from_total_xp(xp):
    return sum(1 for req in lv_xp_req if xp >= req)


def chatwheel_count(m, player, t):
    return sum(
        1 for c in m['chat'] if c['type'] == 'chatwheel' and 'slot' in c and c['slot'] == player and c['time'] < t * 60)


def chat_count(m, player, t):
    return sum(
        1 for c in m['chat'] if c['type'] == 'chat' and 'slot' in c and c['slot'] == player and c['time'] < t * 60)


def purchase_states_array(m, items, t):
    item2id = {}
    for i, j in enumerate(items):
        item2id[j] = i
    r = torch.zeros(10, len(items))
    for i in range(10):
        for j in m['players'][i]['purchase_log']:
            if j['time'] < t * 60 and j['key'] in item2id:
                r[i, item2id[j['key']]] += 1
    return r


def purchase_states_dict(m, items, t):
    r = {}
    for i in items:
        for j in range(5):
            r[f"purchase_{i}_0_{j}"] = 0
        for j in range(5, 10):
            r[f"purchase_{i}_1_{j - 5}"] = 0
    for j in range(5):
        for k in m['players'][j]['purchase_log']:
            if k['time'] < t * 60:
                for h in items:
                    if k['key'] == h:
                        r[f"purchase_{h}_0_{j}"] += 1
    for j in range(5, 10):
        for k in m['players'][j]['purchase_log']:
            if k['time'] < t * 60:
                for h in items:
                    if k['key'] == h:
                        r[f"purchase_{h}_1_{j - 5}"] += 1
    for i in items:
        for j, v in enumerate(sorted([r[f"purchase_{i}_0_{j}"] for j in range(5)])):
            r[f"purchase_{i}_0_{j}"] = v
        for j, v in enumerate(sorted([r[f"purchase_{i}_1_{j - 5}"] for j in range(5, 10)])):
            r[f"purchase_{i}_1_{j}"] = v
    return r


def building_id(s):
    if s.startswith('npc_dota_goodguys_'):
        pos = 0
    elif s.startswith('npc_dota_badguys_'):
        pos = 17
    else:
        raise Exception("tower key " + s + " not recognized")
    if s.endswith('tower1_mid'):
        return pos + 0
    elif s.endswith('tower1_bot'):
        return pos + 1
    elif s.endswith('tower1_top'):
        return pos + 2
    elif s.endswith('tower2_mid'):
        return pos + 3
    elif s.endswith('tower2_bot'):
        return pos + 4
    elif s.endswith('tower2_top'):
        return pos + 5
    elif s.endswith('tower3_mid'):
        return pos + 6
    elif s.endswith('tower3_bot'):
        return pos + 7
    elif s.endswith('tower3_top'):
        return pos + 8
    elif s.endswith('melee_rax_mid'):
        return pos + 9
    elif s.endswith('range_rax_mid'):
        return pos + 10
    elif s.endswith('melee_rax_bot'):
        return pos + 11
    elif s.endswith('range_rax_bot'):
        return pos + 12
    elif s.endswith('melee_rax_top'):
        return pos + 13
    elif s.endswith('range_rax_top'):
        return pos + 14
    elif s.endswith('healers'):
        return pos + 15
    elif s.endswith('tower4'):
        return pos + 16
    elif s.endswith('fort'):
        return -1
    else:
        raise Exception('unrecognized key ' + s)


def building_states_array(m, t):
    r = torch.ones(34)
    r[building_id('npc_dota_goodguys_healers')] += 1
    r[building_id('npc_dota_goodguys_tower4')] += 1
    r[building_id('npc_dota_badguys_healers')] += 1
    r[building_id('npc_dota_badguys_tower4')] += 1
    for o in m['objectives']:
        if o['type'] == 'building_kill' and o['time'] < t * 60:
            bid = building_id(o['key'])
            if bid != -1:
                r[bid] -= 1
    return r


def building_states_dict(m, t):
    r = {}
    blst = ([f"tower{i}_{j}" for i in range(1, 4) for j in ['mid', 'bot', 'top']] +
            [f"{i}_rax_{j}" for i in ['melee', 'range'] for j in ['mid', 'bot', 'top']] +
            ['healers', 'tower4', 'fort'])
    for i in [0, 1]:
        for j in blst:
            r[f"building_{i}_{j}"] = 1
        r[f"building_{i}_tower4"] += 1
        r[f"building_{i}_healers"] += 1
    for o in m['objectives']:
        if o['type'] == 'building_kill' and o['time'] < t * 60:
            if o['key'].startswith('npc_dota_goodguys_'):
                pos = 0
            elif o['key'].startswith('npc_dota_badguys_'):
                pos = 1
            else:
                raise Exception("tower key " + o['key'] + " not recognized")
            for j in blst:
                if o['key'].endswith(j):
                    r[f"building_{pos}_{j}"] -= 1
    return r


def player_kill_count(json_obj, player_slot, time):
    for p in json_obj['players']:
        if p['player_slot'] == player_slot:
            kc = 0
            for k in p['kills_log']:
                if k['time'] < time:
                    kc += 1
            return kc
    raise ValueError('Bad player slot')


def player_death_count(json_obj, player_slot, time):
    target_hero_id = -1
    for p in json_obj['players']:
        if p['player_slot'] == player_slot:
            target_hero_id = p['hero_id']
            break
    if target_hero_id == -1:
        raise ValueError('Bad player slot')
    kc = 0
    for p in json_obj['players']:
        for k in p['kills_log']:
            if k['time'] < time and hero2id[k['key']] == target_hero_id:
                kc += 1
    return kc


def _runes_count(m, player, t):
    r = np.zeros(7)
    for i in m['players'][player]['runes_log']:
        if i['time'] < t * 60:
            for j in range(7):
                if j == i['key']:
                    r[j] += 1
    return r


def rune_states_dict(m, t):
    l0 = np.stack([_runes_count(m, i, t) for i in range(5)]).T
    l1 = np.stack([_runes_count(m, i, t) for i in range(5, 10)]).T
    l0 = np.sort(l0, axis=1)
    l1 = np.sort(l1, axis=1)
    r = {}
    for j in range(7):
        for i in range(5):
            r[f"rune_{j}_0_{i}"] = l0[j, i]
    for j in range(7):
        for i in range(5):
            r[f"rune_{j}_1_{i}"] = l1[j, i]
    return r


class _FileLoader:
    def __init__(self, path_data):
        self.path_data = path_data

    def __call__(self, match_id):
        path_file = os.path.join(self.path_data, f"{match_id}.json")
        lst = []
        with open(path_file, mode='r', encoding='utf-8') as f:
            m = json.load(f)
            # Extract slices of 10%, 20%, ..., 90% of the time of a match.
            for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                t = int(m['duration'] * pct / 60)
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


def load_as_data_frame(path_data, sets=('train', 'valid', 'test')):
    lst = {}
    for s in sets:
        lst[s] = []

    for path_match in tqdm(os.listdir(path_data)):
        if _check_path_match(path_match) in sets and os.path.isfile(os.path.join(path_data, path_match)):
            lst[_check_path_match(path_match)].append(int(path_match[:-5]))

    if 'test' in sets:
        li = lst['test']
    df = {}
    with Pool(12) as pool:
        for s in sets:
            df[s] = pandas.concat(pool.map(_FileLoader(path_data), np.array(lst[s]), chunksize=256), ignore_index=True)
            df[s].region = df[s].region.astype('category')
            del lst[s]
    del lst

    if 'test' in sets:
        return df, li
    else:
        return df


class DotaData(Dataset):
    def __init__(self, path_data, mode, transform=None):
        self.path_data = []
        self.mode = mode
        for path_match in tqdm(os.listdir(path_data)):
            if os.path.isfile(os.path.join(path_data, path_match)):
                if _check_path_match(path_match) == mode:
                    self.path_data.append(os.path.join(path_data, path_match))
        self.transform = transform

    def __len__(self):
        return len(self.path_data)

    def __getitem__(self, index):
        with open(self.path_data[index], mode='r', encoding='utf-8') as f:
            m = json.load(f)
        if self.transform is not None:
            m = self.transform(m)
        return m

    @staticmethod
    def extract_match(m):
        si, sg, e = [], [], []
        hero = torch.LongTensor([m['players'][i]['hero_id'] - 1 for i in range(10)])
        for t in range(len(m['players'][0]['gold_t'])):
            ll = []

            gold = torch.FloatTensor([m['players'][i]['gold_t'][t] for i in range(10)])
            ll.append(gold.view(10, 1))
            xp = torch.FloatTensor([m['players'][i]['xp_t'][t] for i in range(10)])
            ll.append(xp.view(10, 1))
            lh = torch.FloatTensor([m['players'][i]['lh_t'][t] for i in range(10)])
            ll.append(lh.view(10, 1))
            dn = torch.FloatTensor([m['players'][i]['dn_t'][t] for i in range(10)])
            ll.append(dn.view(10, 1))
            lv = torch.FloatTensor([get_lv_from_total_xp(m['players'][i]['xp_t'][t]) for i in range(10)])
            ll.append(lv.view(10, 1))
            chatwheel = torch.FloatTensor([chatwheel_count(m, i, t) for i in range(10)])
            ll.append(chatwheel.view(10, 1))
            chat = torch.FloatTensor([chat_count(m, i, t) for i in range(10)])
            ll.append(chat.view(10, 1))
            pred_vict = torch.FloatTensor([int(m['players'][i]['pred_vict']) for i in range(10)])
            ll.append(pred_vict.view(10, 1))
            purchase = purchase_states_array(m, items_list, t)
            ll.append(purchase)
            rune5 = torch.FloatTensor(
                [sum(1 for r in m['players'][i]['runes_log'] if r['key'] == 5 and r['time'] < t * 60) for i in
                 range(10)])
            ll.append(rune5.view(10, 1))

            gold_ratio = (gold + 1.0) / (gold.sum() + 10.0)
            ll.append(gold_ratio.view(10, 1))
            xp_ratio = (xp + 1.0) / (xp.sum() + 10.0)
            ll.append(xp_ratio.view(10, 1))
            lh_ratio = (lh + 1.0) / (lh.sum() + 10.0)
            ll.append(lh_ratio.view(10, 1))
            dn_ratio = (dn + 1.0) / (dn.sum() + 10.0)
            ll.append(dn_ratio.view(10, 1))

            gold_delta1 = torch.FloatTensor(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in range(10)])
            ll.append(gold_delta1.view(10, 1))
            xp_delta1 = torch.FloatTensor(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(10)])
            ll.append(xp_delta1.view(10, 1))
            gold_delta5 = torch.FloatTensor(
                [m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in range(10)])
            ll.append(gold_delta5.view(10, 1))
            xp_delta5 = torch.FloatTensor(
                [m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(10)])
            ll.append(xp_delta5.view(10, 1))

            gpm = torch.FloatTensor([0 if t == 0 else m['players'][i]['gold_t'][t] / t for i in range(10)])
            ll.append(gpm.view(10, 1))
            xpm = torch.FloatTensor([0 if t == 0 else m['players'][i]['xp_t'][t] / t for i in range(10)])
            ll.append(xpm.view(10, 1))

            si.append(torch.cat(ll, dim=1))

            ll = []

            chatwheel_total = torch.FloatTensor([
                sum(chatwheel_count(m, i, t) for i in range(5)),
                sum(chatwheel_count(m, i, t) for i in range(5, 10))
            ])
            gold_total = torch.FloatTensor([
                sum(m['players'][i]['gold_t'][t] for i in range(5)),
                sum(m['players'][i]['gold_t'][t] for i in range(5, 10))
            ])
            ll.append(gold_total)
            xp_total = torch.FloatTensor([
                sum(m['players'][i]['xp_t'][t] for i in range(5)),
                sum(m['players'][i]['xp_t'][t] for i in range(5, 10))
            ])
            ll.append(xp_total)
            lh_total = torch.FloatTensor([lh[:5].sum(), lh[5:].sum()])
            ll.append(lh_total)
            dn_total = torch.FloatTensor([dn[:5].sum(), dn[5:].sum()])
            ll.append(dn_total)
            rune5_total = torch.FloatTensor([rune5[:5].sum(), rune5[5:].sum()])
            ll.append(rune5_total)
            gold_delta1_total = torch.FloatTensor([
                sum(m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in range(5)),
                sum(m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 1, 0)] for i in range(5, 10))
            ])
            ll.append(gold_delta1_total)
            gold_delta5_total = torch.FloatTensor([
                sum(m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in range(5)),
                sum(m['players'][i]['gold_t'][t] - m['players'][i]['gold_t'][max(t - 5, 0)] for i in range(5, 10))
            ])
            ll.append(gold_delta5_total)
            xp_delta1_total = torch.FloatTensor([
                sum(m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(5)),
                sum(m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 1, 0)] for i in range(5, 10))
            ])
            ll.append(xp_delta1_total)
            xp_delta5_total = torch.FloatTensor([
                sum(m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(5)),
                sum(m['players'][i]['xp_t'][t] - m['players'][i]['xp_t'][max(t - 5, 0)] for i in range(5, 10))
            ])
            ll.append(xp_delta5_total)
            ll = ll + [torch.log1p(t) for t in ll]

            building = building_states_array(m, t)
            ll.append(building)
            ll.append(chatwheel_total)
            pred_vict = torch.FloatTensor([
                sum(m['players'][i]['pred_vict'] for i in range(5)),
                sum(m['players'][i]['pred_vict'] for i in range(5, 10))
            ])
            ll.append(pred_vict)
            gold_variation = torch.FloatTensor([
                stats.variation([m['players'][i]['gold_t'][t] + 1 for i in range(5)]),
                stats.variation([m['players'][i]['gold_t'][t] + 1 for i in range(5, 10)])
            ])
            ll.append(gold_variation)
            xp_variation = torch.FloatTensor([
                stats.variation([m['players'][i]['xp_t'][t] + 1 for i in range(5)]),
                stats.variation([m['players'][i]['xp_t'][t] + 1 for i in range(5, 10)])
            ])
            ll.append(xp_variation)
            region = torch.zeros(25, dtype=torch.float)
            region[m['region'] - 1] = 1.0
            ll.append(region)

            sg.append(torch.cat(ll))

        for i in range(5):
            for j in m['players'][i]['kills_log']:
                for k in range(5, 10):
                    if m['players'][k]['hero_id'] == hero2id[j['key']]:
                        mask = torch.zeros(10, dtype=torch.uint8)
                        mask[[i, k]] = 1
                        e.append((0, j['time'], mask))
                        break
        for i in range(5, 10):
            for j in m['players'][i]['kills_log']:
                for k in range(5):
                    if m['players'][k]['hero_id'] == hero2id[j['key']]:
                        mask = torch.zeros(10, dtype=torch.uint8)
                        mask[[i, k]] = 1
                        e.append((1, j['time'], mask))
                        break
        for i in range(5):
            for j in m['players'][i]['buyback_log']:
                mask = torch.zeros(10, dtype=torch.uint8)
                mask[i] = 1
                e.append((2, j['time'], mask))
        for i in range(5, 10):
            for j in m['players'][i]['buyback_log']:
                mask = torch.zeros(10, dtype=torch.uint8)
                mask[i] = 1
                e.append((3, j['time'], mask))
        for i in range(5):
            for j in m['players'][i]['runes_log']:
                if j['key'] != 5:
                    mask = torch.zeros(10, dtype=torch.uint8)
                    mask[i] = 1
                    e.append((4 + j['key'], j['time'], mask))
        for i in range(5, 10):
            for j in m['players'][i]['runes_log']:
                if j['key'] != 5:
                    mask = torch.zeros(10, dtype=torch.uint8)
                    mask[i] = 1
                    e.append((11 + j['key'], j['time'], mask))
        for i in m['objectives']:
            if i['type'] == 'CHAT_MESSAGE_FIRSTBLOOD':
                mask = torch.zeros(10, dtype=torch.uint8)
                mask[i['slot']] = 1
                e.append((18 if i['slot'] < 5 else 19, i['time'], mask))
            elif i['type'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                mask = torch.zeros(10, dtype=torch.uint8)
                if i['team'] == 2:
                    mask[[0, 1, 2, 3, 4]] = 1
                    e.append((20, i['time'], mask))
                elif i['team'] == 3:
                    mask[[5, 6, 7, 8, 9]] = 1
                    e.append((21, i['time'], mask))
                else:
                    raise Exception('unknown team')
            elif i['type'] == 'CHAT_MESSAGE_AEGIS':
                mask = torch.zeros(10, dtype=torch.uint8)
                mask[i['slot']] = 1
                e.append((22 if i['slot'] < 5 else 23, i['time'], mask))
            elif i['type'] == 'building_kill':
                mask = torch.ones(10, dtype=torch.uint8)
                bid = building_id(i['key'])
                if bid != -1:
                    e.append((24 + bid, i['time'], mask))
        e.append((58, -90, torch.ones(10, dtype=torch.uint8)))
        e = sorted(e, key=lambda x: x[1])

        return si, hero, sg, e, 1 - int(m['radiant_win']), m['duration']

    @staticmethod
    def transform_train(m):
        si, hero, sg, e, y, dur = DotaData.extract_match(m)
        si = si[:128]
        sg = sg[:128]
        e = [ev for ev in e if ev[1] <= len(si) * 60]
        e, e_t, e_mask = zip(*e)
        e = torch.LongTensor(e)
        e_t = torch.LongTensor(e_t)
        e_t.clamp_(max=len(si) * 60 - 1)
        e_mask = torch.stack(e_mask)
        return torch.stack(si), hero, torch.stack(sg), len(si), e, e_t, e_mask, e.size(0), y, dur

    @staticmethod
    def transform_valid(m):
        si, hero, sg, e, y, dur = DotaData.extract_match(m)
        e, e_t, e_mask = zip(*e)
        e = torch.LongTensor(e)
        e_t = torch.LongTensor(e_t)
        e_t.clamp_(max=len(si) * 60 - 1)
        e_mask = torch.stack(e_mask)
        return torch.stack(si), hero, torch.stack(sg), len(si), e, e_t, e_mask, e.size(0), y, dur

    @staticmethod
    def transform_test(m):
        si, hero, sg, e, y, dur = DotaData.extract_match(m)
        e, e_t, e_mask = zip(*e)
        e = torch.LongTensor(e)
        e_t = torch.LongTensor(e_t)
        e_t.clamp_(max=len(si) * 60 - 1)
        e_mask = torch.stack(e_mask)
        return torch.stack(si), hero, torch.stack(sg), len(si), e, e_t, e_mask, e.size(0), y, dur, m['match_id']

    @staticmethod
    def collate_fn(batch):
        batch = sorted(batch, key=lambda item: item[3], reverse=True)
        si_b, hero_b, sg_b, x_len_b, e_b, e_t_b, e_mask_b, e_len_b, y_b, z_b = zip(*batch)
        si = pad_sequence(si_b, batch_first=True)  # Float[bs, max_t, 10, d_si]
        hero = torch.stack(hero_b)  # Long[bs, 10]
        sg = pad_sequence(sg_b, batch_first=True)  # Float[bs, max_t, d_sg]
        x_len = torch.LongTensor(x_len_b)  # Long[bs]
        e = pad_sequence(e_b, batch_first=True)  # Float[bs, max_e]
        e_t = pad_sequence(e_t_b, batch_first=True)  # Float[bs, max_e]
        e_mask = pad_sequence(e_mask_b, batch_first=True)  # Float[bs, max_e, 10]
        e_len = torch.LongTensor(e_len_b)
        y = torch.LongTensor(y_b)  # Long[bs]
        z = torch.FloatTensor(z_b)  # Float[bs]
        return si, hero, sg, x_len, e, e_t, e_mask, e_len, y, z

    @staticmethod
    def collate_fn_with_id(batch):
        batch = sorted(batch, key=lambda item: item[3], reverse=True)
        si_b, hero_b, sg_b, x_len_b, e_b, e_t_b, e_mask_b, e_len_b, y_b, z_b, m_id_b = zip(*batch)
        si = pad_sequence(si_b, batch_first=True)  # Float[bs, max_t, 10, d_si]
        hero = torch.stack(hero_b)  # Long[bs, 10]
        sg = pad_sequence(sg_b, batch_first=True)  # Float[bs, max_t, d_sg]
        x_len = torch.LongTensor(x_len_b)  # Long[bs]
        e = pad_sequence(e_b, batch_first=True)  # Float[bs, max_e]
        e_t = pad_sequence(e_t_b, batch_first=True)  # Float[bs, max_e]
        e_mask = pad_sequence(e_mask_b, batch_first=True)  # Float[bs, max_e, 10]
        e_len = torch.LongTensor(e_len_b)
        y = torch.LongTensor(y_b)  # Long[bs]
        z = torch.FloatTensor(z_b)  # Float[bs]
        m_id = torch.LongTensor(m_id_b)
        return si, hero, sg, x_len, e, e_t, e_mask, e_len, y, z, m_id
