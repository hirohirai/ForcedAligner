#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/04/24

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np

from utils.TextGrid import TextGrid
from utils.Tts import textGrid_to_Tts, tts_to_textGrid
import utils.dsp as dsp

# ログの設定
logger = logging.getLogger(__name__)


def f0_smooth(f0):
    f0s = dsp.median1d(f0, 11)
    f0i = dsp.interpF0(f0s)
    return f0i, f0s


def make_tts(tg_ifs, f0Fn):
    f0data = np.fromfile(f0Fn)
    f0data, f0rdata = f0_smooth(f0data)
    tts = textGrid_to_Tts(TextGrid(tg_ifs), f0=f0data, f0r=f0rdata)
    return tts


def add_pause_subinfo(tts):
    for phr in tts.phrases:
        for wd in phr.words:
            for ix in range(wd.hasPause):
                dur = wd.moras[ix].vlen + wd.moras[ix].clen
                if dur < 0.080:
                    wd.moras[ix].vow = 'sp0'
                elif dur < 0.250:  # 適当。。。
                    wd.moras[ix].vow = 'sp'
                else:
                    wd.moras[ix].vow = 'sp2'


def set_devoice(tts):
    for phr in tts.phrases:
        for wd in phr.words:
            for mr in wd.moras:
                if len(mr.vow) > 0 and mr.vow[0] in 'iuIU' and len(mr.cons)> 0 and mr.cons in ['k','g','s','z','t','c','h','p','s_','t_','c_','f']:  # 'aiueoAIUEO':
                    if mr.F0[2] <= 0.0 and (mr.F0[1] <= 0.0 or mr.F0[3] <= 0.0):
                        mr.vow = mr.vow.upper()
                    else:
                        mr.vow = mr.vow.lower()
                elif len(mr.vow) > 0 and mr.vow[0] in 'AIUEO':  # 'aiueoAIUEO':
                    mr.vow = mr.vow.lower()


def set_accLevel(tts):
    for phr in tts.phrases:
        for wd in phr.words:
            if 0 < wd.acc_down < len(wd.moras)-wd.hasPause:
                hif0 = []
                lof0 = []
                st = wd.hasPause + 1 if wd.acc_down > 1 else wd.hasPause
                for mr in wd.moras[st:wd.acc_down+wd.hasPause]:
                    hif0.extend(mr.F0)
                for mr in wd.moras[wd.acc_down+wd.hasPause:]:
                    lof0.extend(mr.F0)
                hF0 = np.mean(np.abs(hif0[-5:-2]))
                lF0 = np.mean(np.abs(lof0[2:5]))
                if lF0 / hF0 < 0.4:
                    wd.acc_level = 2
                elif lF0 / hF0 > 0.8:
                    wd.acc_level = 0
                else:
                    wd.acc_level = 1


def set_wordDiv(tts):
    lawdf0 = None # [mr2nd, mracc]
    for phr in tts.phrases:
        for wd in phr.words:
            if wd.acc_down == 1 or len(wd.moras) == wd.hasPause+1:
                mr2nd = np.mean(np.abs(wd.moras[wd.hasPause].F0[1:4]))
                mracc = mr2nd
            else:
                mr2nd = np.mean(np.abs(wd.moras[wd.hasPause + 1].F0[1:4]))
                if wd.acc_down > 0:
                    mracc = np.mean(np.abs(wd.moras[wd.hasPause + wd.acc_down].F0[1:4]))
                else:
                    mracc = np.mean(np.abs(wd.moras[-1].F0[1:4]))

            if lawdf0:
                if mr2nd / lawdf0[0] > 1.2:
                    wd.bound_div = 2
                elif mr2nd / lawdf0[0] > 0.8:
                    wd.bound_div = 1
                else:
                    wd.bound_div = 0
            else:
                wd.bound_div = 0

            lawdf0 = [mr2nd, mracc]



def set_kanaSubInfo(f0Fn, tg_ifs, tg_ofs, mode):
    tts = make_tts(tg_ifs, f0Fn)
    if 'P' in mode:
        add_pause_subinfo(tts)
    if 'D' in mode:
        set_devoice(tts)
    if 'A' in mode:
        set_accLevel(tts)
    if 'W' in mode:
        set_wordDiv(tts)
    logger.info(str(tts))
    tg = tts_to_textGrid(tts)
    tg.addStEd()
    print(tg, file=tg_ofs)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('f0')
    parser.add_argument('--inFn', '-i', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--outFn', '-o', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--mode', '-m' , default='PD', help='PD')

    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--log', default='')
    args = parser.parse_args()

    if args.debug:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

    set_kanaSubInfo(args.f0, args.inFn, args.outFn, args.mode)
