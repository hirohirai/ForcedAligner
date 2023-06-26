#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/04/14

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

import utils.TextGrid
import utils.Tts


# ログの設定
logger = logging.getLogger(__name__)

def textGrid2lbl(tg, width):
    lbls = []
    st = 0
    for ph in tg.get_phoneme():
        ed = round(ph.xmax / width)
        for ii in range(st, ed):
            lbls.append(ph.text)
        st = ed
    return lbls

def calc_prob_lbl(lbl1, lbl2):
    correct = 0
    ph_correct = {}
    ph_num = {}
    for ix, l1 in enumerate(lbl1):
        if l1 in ph_num:
            ph_num[l1] += 1
        else:
            ph_num[l1] = 1
            ph_correct[l1] = 0
        if l1==lbl2[ix]:
            correct += 1
            ph_correct[l1] += 1
    ph_prob = {}
    for kk in ph_num.keys():
        ph_prob[kk] = ph_correct[kk] / ph_num[kk]

    return correct / len(lbl1), ph_prob

def main(args):
    tg1 = utils.TextGrid.TextGrid(args.file1)
    tts = utils.Tts.textGrid_to_Tts(tg1)
    tts.div_all()
    tg1 = utils.Tts.tts_to_textGrid(tts)
    lbl1 = textGrid2lbl(tg1, args.width)

    tg2 = utils.TextGrid.TextGrid(args.file2)
    tts = utils.Tts.textGrid_to_Tts(tg2)
    tts.div_all()
    tg2 = utils.Tts.tts_to_textGrid(tts)
    lbl2 = textGrid2lbl(tg2, args.width)


    prob, ph_prob = calc_prob_lbl(lbl1, lbl2)
    print(prob)
    for kk in ph_prob.keys():
        print(f'{kk} {ph_prob[kk]:0.4}')



if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
    parser.add_argument('--width', type=float, default=0.005)
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
