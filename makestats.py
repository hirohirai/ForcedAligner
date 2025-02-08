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

import utils.TextGrid
from utils.var import DataSetList, DataIdList
from utils.paths import mk_spk_dirs
from utils.world_cof import WorldCof, StatsCof

TESTMAIN=False


def preprocess(spk, f_ids, stats_bdir, stats_file, feats_dir, mgcP, capP, ffthn, mfccn, DoFlg):
    wcof_all = WorldCof(mgcP=mgcP, capP=capP)
    wcof_all.clear()
    scof_all = StatsCof(param_n=ffthn)
    scof_all.clear()
    mcof_all = StatsCof(param_n=mfccn)
    mcof_all.clear()
    for fid in f_ids:
        try:
            if 'mgc' in DoFlg:
                fn=f'{feats_dir}/{spk}/{fid}.npz'
                tmp = np.load(fn)
                world = tmp['in_feat']

                wcof = WorldCof(mgcP=mgcP, capP=capP)
                wcof.clear()
                wcof.set_dat(world[:,0], world[:,1:mgcP+1], world[:, -capP:])
                wcof.calc_stat(True)

                wcof_all.add(wcof)

            if 'spc' in DoFlg:
                fn=f'{feats_dir}/{spk}/{fid}_stft.npz'
                tmp = np.load(fn)
                stft = tmp['in_feat']

                scof = StatsCof(param_n=ffthn)
                scof.clear()
                scof.set_dat(stft)
                scof.calc_stat(True)

                scof_all.add(scof)

            if 'mfcc' in DoFlg:
                fn=f'{feats_dir}/{spk}/{fid}_mfcc.npz'
                tmp = np.load(fn)
                mfcc = tmp['in_feat']

                mcof = StatsCof(param_n=mfccn)
                mcof.clear()
                mcof.set_dat(mfcc)
                mcof.calc_stat(True)

                mcof_all.add(mcof)


        except Exception as e:
            logger.error(f'{e} {spk}:{fid}')

    if 'mgc' in DoFlg:
        wcof_all.calc_stat(True)
        fn = f'{stats_bdir}/{spk}/{stats_file[0]}'
        wcof_all.save(fn)

    if 'spc' in DoFlg:
        scof_all.calc_stat(True)
        fn = f'{stats_bdir}/{spk}/{stats_file[1]}'
        scof_all.save(fn)

    if 'mfcc' in DoFlg:
        mcof_all.calc_stat(True)
        fn = f'{stats_bdir}/{spk}/{stats_file[2]}'
        mcof_all.save(fn)

    return wcof_all, scof_all, mcof_all


if TESTMAIN == False:

    @hydra.main(version_base='1.2', config_path="conf", config_name='config')
    def my_app(cfg: DictConfig) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\n"+OmegaConf.to_yaml(cfg))


        file_ids = DataIdList(filename=f'{cfg.data.base_dir}/dataset_input.csv')
        logging.warning(f'\n{len(file_ids)} tgrid files found in "{cfg.input.tgrid_path}"')

        stats_bdir = f'{cfg.data.base_dir}/{cfg.data.stats_dir}'

        mk_spk_dirs(stats_bdir, file_ids)

        fid_dic = file_ids.get_spkid_dic()
        wcof_all = WorldCof(mgcP=cfg.data.mgc_order, capP=cfg.input.cap_order)
        wcof_all.clear()
        scof_all = StatsCof(param_n=int(cfg.input.fftn/2+1))
        scof_all.clear()
        mcof_all = StatsCof(param_n=13)
        mcof_all.clear()
        for spk in fid_dic.keys():
            try:
                world_cof, stft_cof, mfcc_cof = preprocess(spk, fid_dic[spk], stats_bdir, cfg.data.stats_file, f'{cfg.data.base_dir}/{cfg.data.input_dir}', cfg.data.mgc_order, cfg.input.cap_order, int(cfg.input.fftn/2+1), cfg.input.mfccn, cfg.data.use_stats)
                wcof_all.add(world_cof)
                scof_all.add(stft_cof)
                mcof_all.add(mfcc_cof)
            except Exception as e:
                logging.error(f'{spk} {e}')

        if 'mgc' in cfg.data.use_stats:
            fn = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/world'
            wcof_all.calc_stat()
            wcof_all.save(fn)
        if 'spc' in cfg.data.use_stats:
            fn = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/stft'
            scof_all.calc_stat()
            scof_all.save(fn)
        if 'mfcc' in cfg.data.use_stats:
            fn = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/mfcc'
            mcof_all.calc_stat()
            mcof_all.save(fn)


else:
    def my_app():
        feats_bdir = 'data/in_feats'
        stats_bdir = 'data/testdir'
        os.makedirs(stats_bdir, exist_ok=True)

        wcof_all = WorldCof(mgcP=25, capP=2)
        wcof_all.clear()
        scof_all = StatsCof(param_n=129)
        scof_all.clear()
        mcof_all = StatsCof(param_n=13)
        mcof_all.clear()

        spk = 'M207'
        f_ids = ('A01', 'B43')
        os.makedirs(f'{stats_bdir}/{spk}', exist_ok=True)
        # preprocess(spk, f_ids, stats_bdir, stats_file, feats_dir, mgcP, capP, ffthn)

        wcof, scof, mcof = preprocess(spk, f_ids, stats_bdir, ['world.npz', 'stft.npz', 'mfcc.npz'], feats_bdir, 25, 2, 129, 13)
        wcof_all.add(wcof)
        scof_all.add(scof)
        mcof_all.add(mcof)

        spk = 'M101'
        f_ids = ('A01', 'A02')
        os.makedirs(f'{stats_bdir}/{spk}', exist_ok=True)
        wcof, scof, mcof = preprocess(spk, f_ids, stats_bdir, ['world.npz', 'stft.npz', 'mfcc.npz'], feats_bdir, 25, 2, 129, 13)
        wcof_all.add(wcof)
        scof_all.add(scof)
        mcof_all.add(mcof)

        fn = f'{stats_bdir}/world'
        wcof_all.calc_stat()
        wcof_all.save(fn)

        fn = f'{stats_bdir}/stft'
        scof_all.calc_stat()
        scof_all.save(fn)

        fn = f'{stats_bdir}/mfcc'
        mcof_all.calc_stat()
        mcof_all.save(fn)

if __name__ == "__main__":
    my_app()

