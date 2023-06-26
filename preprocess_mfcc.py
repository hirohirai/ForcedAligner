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


def add_phone_force(wix, LastIx, wrate, ed, ophn, lbl):
    if wix >= LastIx or wix * wrate + wrate/2 < ed:
        ophn[-1] = rom_to_id[lbl]
    else:
        ophn.append(rom_to_id[lbl])
        wix += 1
    return wix

def preprocess(f_id, tbdir, wvbdir, feats_bdir, target_bdir, wrate=0.005, subdir=None):
    try:
        fnb = f'{f_id[0]}/{f_id[1][0]}/{f_id[1]}' if subdir else f'{f_id[0]}/{f_id[1]}'
        tgfn = f'{tbdir}/{fnb}.TextGrid'
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

        [wv, sr] = librosa.load(wvfn, sr=16000)
        wvpe = librosa.effects.preemphasis(wv)
        mfcc = librosa.feature.mfcc(y=wvpe, sr=sr, n_mfcc=13, hop_length=80)
        mfcc = mfcc.T

        featfnb = f'{feats_bdir}/{f_id[0]}/{f_id[1]}_mfcc'
        targetfnb = f'{target_bdir}/{f_id[0]}/{f_id[1]}'

        sttime = tts.phrases[0].words[0].moras[0].get_st()
        edtime = tts.phrases[-1].words[-1].moras[-1].get_ed()
        StIx = int((sttime - 0.050) / wrate)
        LastIx = int((edtime + 0.050) / wrate)

        tgt = np.load(f'{targetfnb}.npz')
        if tgt['target'].shape[0] < LastIx-StIx:
            LastIx = tgt['target'].shape[0] + StIx

        mfcc = mfcc[StIx:LastIx]
        time = np.arange(StIx, LastIx) * wrate

        if tgt['target'].shape[0] != LastIx-StIx:
            logger.error(f"Size Error {tgt['target'].shape}, {StIx}, {LastIx}, ID={f_id[0]}/{f_id[1]}")
        else:
            np.savez_compressed(featfnb, in_feat=mfcc)
            np.savez_compressed(targetfnb, target=tgt['target'], time=time)
    except Exception as e:
        logging.error(f'{f_id} {e}')



if TESTMAIN == False:

    @hydra.main(version_base='1.2', config_path="conf", config_name='config')
    def my_app(cfg: DictConfig) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\n"+OmegaConf.to_yaml(cfg))

        os.makedirs(cfg.data.base_dir, exist_ok=True)

        spk_ptn = cfg.input.spk_ptn + cfg.input.subdir if cfg.input.subdir else cfg.input.spk_ptn
        file_ids = DataIdList(cfg.input.tgrid_path, spk_ptn, '.TextGrid')
        file_ids_mgc = DataIdList(cfg.input.world_path, spk_ptn, '.mgc')
        file_ids_wav = DataIdList(cfg.input.wav_path, spk_ptn, '.wav')
        file_ids = file_ids.intersection(file_ids_wav)
        file_ids = file_ids.intersection(file_ids_mgc)
        logging.warning(f'\n{len(file_ids)} tgrid files found in "{cfg.input.tgrid_path}"')

        tbdir = cfg.input.tgrid_path
        wv_bdir = cfg.input.wav_path
        feats_bdir = f'{cfg.data.base_dir}/{cfg.data.input_dir}'
        target_bdir = f'{cfg.data.base_dir}/{cfg.data.target_dir}'

        mk_spk_dirs(feats_bdir, file_ids)
        mk_spk_dirs(target_bdir, file_ids)

        file_ids.save(f'{cfg.data.base_dir}/dataset_input.csv')

        with ProcessPoolExecutor(cfg.N_jobs) as executor:
            futures = [
                executor.submit(
                    preprocess, f_id, tbdir, wv_bdir, feats_bdir, target_bdir, subdir=cfg.input.subdir
                )
                for f_id in file_ids
            ]
            for future in tqdm(futures):
                future.result()

else:
    def my_app():
        logging.basicConfig(level=logging.WARNING)
        f_id = ('F119', 'A34')
        tbdir = '/home/hirai/work_local/Speech/DBS_/atr/TextGrid'
        wv_bdir = '/home/hirai/work_local/Speech/DBS_/atr/wav'
        feats_bdir = 'data/testdir/in_feats'
        target_bdir = 'data/testdir/targets'
        os.makedirs(feats_bdir, exist_ok=True)
        os.makedirs(target_bdir, exist_ok=True)
        preprocess(f_id, tbdir, wv_bdir, feats_bdir, target_bdir)


if __name__ == "__main__":
    my_app()

