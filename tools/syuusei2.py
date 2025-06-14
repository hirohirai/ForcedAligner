#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/12/08

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

from utils.TextGrid import TextGrid
from utils.Tts import Tts, textGrid_to_Tts, tts_to_textGrid, set_default_join, set_default_cl_z

# ログの設定
logger = logging.getLogger(__name__)

'''
# frameがb始まりになってなかった。
def main(args):
    tg1 = TextGrid(args.file1)
    tg2 = TextGrid(args.file2)

    if tg1.get_frame(0).text.startswith('b'):
        for fr in tg2.get_frame():
            fr.text = 'b' + fr.text[1:]

        with open(args.ofile, 'w') as ofs:
            print(tg2, file=ofs)
'''


def set_xmax_phoneme(tg):
    phs = tg.get_phoneme()
    for ix in range(len(phs)-1):
        if phs[ix].xmax != phs[ix+1].xmin:
            phs[ix].xmax = phs[ix+1].xmin

    return tg


def reset_bound_div(tts):
    for pr in tts.phrases:
        for wrd in pr.words:
            if wrd.hasPause:
                wrd.bound_div = 2
            else:
                wrd.bound_div = 0


def reset_sp(tts):
    for mr in tts.moras:
        if mr.vow in ['sp0', 'sp1', 'sp2']:
            mr.vow = 'sp'


def main(args):
    tg = TextGrid(args.file1)
    tts = textGrid_to_Tts(tg, force_clJ=True)

    # ポーズの直後だけにフレーズ境界を入れる
    #reset_bound_div(tts)
    # sp のみに変更
    #reset_sp(tts)

    set_default_cl_z(tts)
    set_default_join(tts)

    tgo = tts_to_textGrid(tts)
    #tgo = set_xmax_phoneme(tgo)
    tgo.correct_times()

    # trans層の始端、終端はずれていても採用する
    sted = tg.getStEd('trans')
    tr = tgo.get_trans()
    tr[1].xmin = sted[0]
    tr[-1].xmin = sted[1]

    with open(args.ofile, 'w') as ofs:
        print(tgo, file=ofs)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('ofile')
    # parser.add_argument('-s', '--opt_str', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
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

    main(args)
