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

from utils.TextGrid import TextGrid
from utils.var import DataSetList, DataIdList
import utils.dur_stats
import utils.Tts


TESTMAIN=False


def preprocess(f_ids, tg_bdir):
    durs = utils.dur_stats.DurStats()
    tgfn = f'{tg_bdir}/{f_ids[0]}/{f_ids[1][0]}/{f_ids[1]}.TextGrid'
    tg = TextGrid(tgfn)
    tts = utils.Tts.textGrid_to_Tts(tg)
    tts.div_all()
    tgo = utils.Tts.tts_to_textGrid(tts)
    for phn in tgo.get_phoneme():
        if phn.text == '#':
            durs.add('sp', phn.xmax - phn.xmin)
        else:
            durs.add(phn.text, phn.xmax - phn.xmin)

    return durs


if TESTMAIN == False:

    @hydra.main(version_base='1.2', config_path="conf", config_name='config_atrdb')
    def my_app(cfg: DictConfig) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("\n"+OmegaConf.to_yaml(cfg))


        file_ids = DataIdList(filename=f'{cfg.data.base_dir}/dataset_input.csv')
        logging.warning(f'\n{len(file_ids)} tgrid files found in "{cfg.input.tgrid_path}"')

        tg_bdir = cfg.input.tgrid_path
        durs_all = utils.dur_stats.DurStats()

        for f_ids in file_ids:
            durs = preprocess(f_ids, tg_bdir)
            durs_all.merge(durs)

        durl = utils.dur_stats.DurLogLikelihood(durs_all)

        dicfn = f'{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.dur_dic}'
        durl.save(dicfn)

else:
    def my_app():


        f_ids = ('F101', 'B43')

        durs = preprocess(f_ids, '../DBS_/atr/TextGrid')



if __name__ == "__main__":
    my_app()

