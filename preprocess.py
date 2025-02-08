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
import librosa

import utils.TextGrid
import utils.Tts
from utils.var import DataSetList, DataIdList
from utils.paths import mk_spk_dirs
from utils.world_cof import WorldCof, StatsCof
from utils.text.rtMRI import rom_to_id

TESTMAIN=False


def delta(feats):
    d_feats = np.zeros(feats.shape)
    d_feats[1:-1] = (feats[2:] - feats[:-2])/2
    d_feats[0] = feats[1] - feats[0]
    d_feats[-1] = feats[-1] - feats[-2]
    dd_feats = np.zeros(feats.shape)
    dd_feats[1:-1] = (d_feats[2:] - d_feats[:-2])/2
    dd_feats[0] = d_feats[1] - d_feats[0]
    dd_feats[-1] = d_feats[-1] - d_feats[-2]
    return np.hstack([feats, d_feats, dd_feats])


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


def add_phone_force(wix, LastIx, wrate, ed, ophn, lbl):
    if wix >= LastIx or wix * wrate + wrate/2 < ed:
        ophn[-1] = rom_to_id[lbl]
    else:
        ophn.append(rom_to_id[lbl])
        wix += 1
    return wix


def preprocess(f_id, tbdir, world_bdir, wvbdir, feats_bdir, target_bdir, mgcPo, wrate=0.005, subdir=None):
    fnb = f'{f_id[0]}/{f_id[1][0]}/{f_id[1]}' if subdir else f'{f_id[0]}/{f_id[1]}'
    f_id_wv = f_id[0].split('_')[0]
    fnb_wv = f'{f_id_wv}/{f_id[1][0]}/{f_id[1]}' if subdir else f'{f_id_wv}/{f_id[1]}'
    tgfn = f'{tbdir}/{fnb_wv}.TextGrid'
    wvfn = f'{wvbdir}/{fnb}.wav'

    logger.debug(f'{tgfn}')
    tg = utils.TextGrid.TextGrid(tgfn)
    kana_sent = utils.text.kana.KanaSent()
    kana = tg.get_jeitaKana()
    kana_sent.set_kana(kana)
    tts = utils.Tts.Tts()
    tts.from_kanaSent(kana_sent)
    tts.set_time(tg)

    tts.xmin = tg.xmin
    tts.xmax = tg.xmax

    tts.lv_HtoXH()

    world = np.load(f'{world_bdir}/{fnb}.npz')

    [wv, sr] = librosa.load(wvfn, sr=10000)
    r = np.random.randn(len(wv)) * 1e-5
    # r[wv!=0.0] = 0
    stft = np.log(np.abs(librosa.stft(wv+r, n_fft=256, hop_length=50))).T

    featfnb = f'{feats_bdir}/{f_id[0]}/{f_id[1]}'
    featfnb2 = f'{feats_bdir}/{f_id[0]}/{f_id[1]}_stft'
    targetfnb = f'{target_bdir}/{f_id[0]}/{f_id[1]}'

    sttime = tts.phrases[0].words[0].moras[0].get_st()
    edtime = tts.phrases[-1].words[-1].moras[-1].get_ed()
    ophn =[]
    StIx = int((sttime - 0.050) / wrate)
    wix = StIx
    LastIx = int((edtime + 0.050) / wrate)
    if world['f0'].shape[0] < LastIx:
        LastIx = world['f0'].shape[0]
    for phr in tts.phrases:
        for wd in phr.words:
            for mr in wd.moras:
                ed = mr.st - 0.0000001
                while wix < LastIx and wix * wrate < ed:
                    if wix * wrate > sttime:
                        logger.error(f'{f_id} SKIP data {wix*wrate} {sttime} {wix} {StIx} {ed} {mr}')
                    ophn.append(0)
                    wix += 1
                if mr.CL_len > 0:
                    ed += mr.CL_len
                    flg = True
                    while wix < LastIx and wix * wrate < ed:
                        ophn.append(rom_to_id['<cl>'])
                        wix += 1
                        flg = False
                    if flg:
                        wix = add_phone_force(wix, LastIx, wrate, ed, ophn, '<cl>')

                if mr.clen > 0:
                    ed += mr.clen
                    flg = True
                    while wix < LastIx and wix*wrate < ed:
                        ophn.append(rom_to_id[mr.cons])
                        wix += 1
                        flg = False
                    if flg:
                        wix = add_phone_force(wix, LastIx, wrate, ed, ophn, mr.cons)
                if mr.vlen > 0:
                    ed += mr.vlen
                    flg = True
                    while wix < LastIx and wix*wrate < ed:
                        ophn.append(rom_to_id[mr.vow])
                        wix += 1
                        flg = False
                    if flg:
                        wix = add_phone_force(wix, LastIx, wrate, ed, ophn, mr.vow)

                if mr.J_len> 0:
                    ed += mr.J_len
                    flg = True
                    while wix < LastIx and wix*wrate < ed:
                        ophn.append(rom_to_id['<j>'])
                        wix += 1
                        flg = False
                    if flg:
                        wix = add_phone_force(wix, LastIx, wrate, ed, ophn, '<j>')
    for ix in range(wix, LastIx):
        ophn.append(0)


    targets = np.array(ophn, dtype=int)
    in_feats = np.hstack([world['f0'][:, np.newaxis], world['mgc'][:,:mgcPo], world['cap']])
    #in_feats = wcof.encode(f0, mgc, cap)
    in_feats = in_feats[StIx:LastIx]
    stft = stft[StIx:LastIx]
    time = np.arange(StIx, LastIx) * wrate

    #in_feats_d = delta(in_feats)
    #in_feats_d = np.hstack([in_feats_d, stft])
    #in_feats_de = expand(in_feats_d)

    #in_feats_de = in_feats_de[3:-3]
    #targets = targets[3:-3]

    np.savez_compressed(featfnb, in_feat=in_feats)
    np.savez_compressed(featfnb2, in_feat=stft)
    np.savez_compressed(targetfnb, target=targets, time=time)


if TESTMAIN == False:

    @hydra.main(version_base='1.2', config_path="conf", config_name='config')
    def my_app(cfg: DictConfig) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\n"+OmegaConf.to_yaml(cfg))

        os.makedirs(cfg.data.base_dir, exist_ok=True)

        spk_ptn = cfg.input.spk_ptn + cfg.input.subdir if cfg.input.subdir else cfg.input.spk_ptn
        file_ids = DataIdList(cfg.input.tgrid_path, spk_ptn, '.TextGrid')
        file_ids_mgc = DataIdList(cfg.input.world_path, spk_ptn, '.npz')
        file_ids = file_ids.intersection(file_ids_mgc)
        logging.warning(f'\n{len(file_ids)} tgrid files found in "{cfg.input.tgrid_path}"')

        tbdir = cfg.input.tgrid_path
        world_bdir = cfg.input.world_path
        world_f0_bdir = cfg.input.world_f0_path
        wv_bdir = cfg.input.wav_path
#        stats_file = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[0]}'
#        stats_file_stft = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[1]}'
#        wcof = WorldCof(stats_file)
#        scof = StftCof(stats_file_stft)
        feats_bdir = f'{cfg.data.base_dir}/{cfg.data.input_dir}'
        target_bdir = f'{cfg.data.base_dir}/{cfg.data.target_dir}'
        mgcPo = cfg.data.mgc_order

        mk_spk_dirs(feats_bdir, file_ids)
        mk_spk_dirs(target_bdir, file_ids)

        file_ids.save(f'{cfg.data.base_dir}/dataset_input.csv')

        with ProcessPoolExecutor(cfg.N_jobs) as executor:
            futures = [
                executor.submit(
                    preprocess, f_id, tbdir, world_bdir, wv_bdir, feats_bdir, target_bdir, mgcPo,
                    subdir=cfg.input.subdir
                )
                for f_id in file_ids
            ]
            for future in tqdm(futures):
                future.result()

else:
    def my_app():
        logging.basicConfig(level=logging.WARNING)
        f_id = ('F117_2', 'C25')
        tbdir = '/home/hirai/work_local/Speech/DBS_/atr/TextGrid'
        world_bdir = '/home/hirai/work_local/Speech/DBS_/atr/nr_world'
        world_f0_bdir = '/home/hirai/work_local/Speech/DBS_/atr/nr_world'
        wv_bdir = '/home/hirai/work_local/Speech/DBS_/atr/nr_wav'
        # stats_file = 'data/stats/world.npz'
        # stats_file_stft = 'data/stats/stft.npz'
        # wcof = WorldCof(stats_file)
        # scof = StatsCof(stats_file_stft)
        feats_bdir = 'data/testdir/in_feats'
        target_bdir = 'data/testdir/targets'
        mgcPo = 25
        subdir = '.'
        os.makedirs(feats_bdir, exist_ok=True)
        os.makedirs(target_bdir, exist_ok=True)
        preprocess(f_id, tbdir, world_bdir, wv_bdir, feats_bdir, target_bdir, mgcPo, subdir=subdir)


if __name__ == "__main__":
    my_app()

