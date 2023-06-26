#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

    TextGrid から　FullContextファイルを作成する
"""

import sys, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
# ログの設定
logger = logging.getLogger(__name__)


import numpy as np
from tqdm import tqdm

from utils.var import DataSetList, DataIdList
from utils.paths import mk_spk_dirs
from utils.world_cof import WorldCof, StatsCof

TESTMAIN=False

'''
def expand(feats, dt=[1,3]):
    if len(dt) == 1:
        l1 = np.zeros(feats.shape)
        n1 = np.zeros(feats.shape)
        l1[dt[0]:] = feats[:-dt[0]]
        n1[:-dt[0]] = feats[dt[0]:]

        return np.hstack([l1, feats, n1])

    elif len(dt) == 2:
        l2 = np.zeros(feats.shape)
        l1 = np.zeros(feats.shape)
        n1 = np.zeros(feats.shape)
        n2 = np.zeros(feats.shape)
        l1[dt[0]:] = feats[:-dt[0]]
        l2[dt[1]:] = feats[:-dt[1]]
        n1[:-dt[0]] = feats[dt[0]:]
        n1[:-dt[1]] = feats[dt[1]:]

        return np.hstack([l2, l1, feats, n1, n2])

    return None
'''
def expand(feats, mae=[1,2,], usiro=[1,2,]):
    ofts = []
    for ix in mae:
        m_ = np.pad(feats[:-ix], [[ix, 0], [0, 0]], 'edge')
        ofts.insert(0, m_)
    ofts.append(feats)
    for ix in usiro:
        m_ = np.pad(feats[ix:], [[0, ix], [0, 0]], 'edge')
        ofts.append(m_)
    ofeats = np.array(ofts, dtype=np.float32).transpose(1, 0, 2)

    return ofeats

def conv_data(wcof, scof, mcof, mgc, spc, mfcc, use_stats):
    if 'mgc' in use_stats:
        mgc = wcof.encode(mgc)
        mgc_s = expand(mgc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    else:
        mgc_s = None
    if 'spc' in use_stats:
        spc = scof.encode(spc)
        spc_s = expand(spc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    else:
        spc_s = None
    if 'mfcc' in use_stats:
        mfcc = mcof.encode(mfcc)
        mfcc_s = expand(mfcc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    else:
        mfcc_s = None

    return mgc_s, spc_s, mfcc_s

def grep_data(targets_, mgc_s_, spc_s_, mfcc_s_):
    targets = np.array([], dtype=np.int64)
    mgc_s = None
    spc_s = None
    mfcc_lst = []
    for ix in range(targets_.shape[0]):
        if targets_[ix] < 6:
            targets = np.append(targets, targets_[ix])
            mfcc_lst.append(mfcc_s_[ix])

    return targets, mgc_s, spc_s, np.array(mfcc_lst)


def preprocess(f_id, wcof, scof, mcof, feats_bdir, target_bdir, ofeats_bdir, otarget_bdir, use_stats):
    try:
        if 'mgc' in use_stats:
            indfn = f'{feats_bdir}/{f_id[0]}/{f_id[1]}.npz'
            in1 = np.load(indfn)
            in1 = in1['in_feat']
        else:
            in1 = None
        if 'spc' in use_stats:
            indfn = f'{feats_bdir}/{f_id[0]}/{f_id[1]}_stft.npz'
            in2 = np.load(indfn)
            in2 = in2['in_feat']
        else:
            in2 = None
        if 'mfcc' in use_stats:
            indfn = f'{feats_bdir}/{f_id[0]}/{f_id[1]}_mfcc.npz'
            in3 = np.load(indfn)
            in3 = in3['in_feat']
        else:
            in3 = None

        tgtfn = f'{target_bdir}/{f_id[0]}/{f_id[1]}.npz'
        targets = np.load(tgtfn)['target']
        mgc_s, spc_s, mfcc_s= conv_data(wcof, scof, mcof, in1, in2, in3, use_stats)
        # targets, mgc_s, spc_s, mfcc_s = grep_data(targets, mgc_s, spc_s, mfcc_s)

        ofeatfn = f'{ofeats_bdir}/{f_id[0]}/{f_id[1]}.npz'
        np.savez_compressed(ofeatfn, mgc=mgc_s, spc=spc_s, mfcc=mfcc_s)

        otgtfn = f'{otarget_bdir}/{f_id[0]}/{f_id[1]}.npz'
        np.savez_compressed(otgtfn, target=targets)
    except Exception as e:
        logging.error(f'{f_id}: {e}')

if TESTMAIN == False:

    @hydra.main(version_base='1.2', config_path="conf", config_name='config')
    def my_app(cfg: DictConfig) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\n"+OmegaConf.to_yaml(cfg))

        file_ids = DataIdList(filename=f'{cfg.data.base_dir}/dataset_input.csv')
        stats_file = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[0]}'
        stats_file_stft = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[1]}'
        stats_file_mfcc = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[2]}'
        wcof = WorldCof(stats_file)
        scof = StatsCof(stats_file_stft)
        mcof = StatsCof(stats_file_mfcc)
        feats_bdir = f'{cfg.data.base_dir}/{cfg.data.input_dir}'
        target_bdir = f'{cfg.data.base_dir}/{cfg.data.target_dir}'
        ofeats_bdir = f'{cfg.data.train_dir}/{cfg.data.input_dir}'
        otarget_bdir = f'{cfg.data.train_dir}/{cfg.data.target_dir}'

        mk_spk_dirs(ofeats_bdir, file_ids)
        mk_spk_dirs(otarget_bdir, file_ids)

        file_ids.save(f'{cfg.data.base_dir}/dataset_input.csv')

        with ProcessPoolExecutor(cfg.N_jobs) as executor:
            futures = [
                executor.submit(
                    preprocess, f_id, wcof, scof, mcof, feats_bdir, target_bdir, ofeats_bdir, otarget_bdir, cfg.data.use_stats
                )
                for f_id in file_ids
            ]
            for future in tqdm(futures):
                future.result()

else:
    def my_app():
        logging.basicConfig(level=logging.WARNING)
        f_id = ('F119', 'A34')
        stats_file = 'data/stats/world.npz'
        stats_file_stft = 'data/stats/stft.npz'
        wcof = WorldCof(stats_file)
        scof = StftCof(stats_file_stft)
        feats_bdir = 'data/in_feats'
        target_bdir = 'data/targets'
        ofeats_bdir = 'data/testdir/in_feats'
        otarget_bdir = 'data/testdir/targets'
        os.makedirs(f'{ofeats_bdir}/{f_id[0]}', exist_ok=True)
        os.makedirs(f'{otarget_bdir}/{f_id[0]}', exist_ok=True)

        preprocess(f_id, wcof, scof, feats_bdir, target_bdir, ofeats_bdir, otarget_bdir)


if __name__ == "__main__":
    my_app()

