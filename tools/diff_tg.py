#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2024/06/10

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import os.path

import difflib

from utils.TextGrid import TextGrid
# ログの設定
logger = logging.getLogger(__name__)


def Print_(fnb, tg, st, ed):
    if st >= len(tg.get_phoneme()):
        st = len(tg.get_phoneme()) -1

    if ed-1 >= len(tg.get_phoneme()):
        ed = len(tg.get_phoneme())

    print(fnb, tg.get_phoneme(st).xmin, tg.get_phoneme(ed-1).xmax)

def diff_phoneme(args):
    fnb = os.path.splitext(os.path.basename(args.file1))[0]
    tg1 = TextGrid(args.file1)
    tg2 = TextGrid(args.file2)
    tg1_phs = [ph.text for ph in tg1.get_phoneme()]
    tg2_phs = [ph.text for ph in tg2.get_phoneme()]

    sdiff = difflib.SequenceMatcher(None, tg1_phs, tg2_phs)
    lpos_a = 0
    lpos_b = 0
    for mt in sdiff.get_matching_blocks():
        if lpos_a != mt.a:
            Print_(fnb, tg1, lpos_a, mt.a)
        elif lpos_b != mt.b:
            Print_(fnb, tg1, lpos_a, lpos_a+1)
        lpos_a = mt.a + mt.size
        lpos_b = mt.b + mt.size


def diff_sp_time(args):
    fnb = os.path.splitext(os.path.basename(args.file1))[0]
    tg1 = TextGrid(args.file1)
    tg2 = TextGrid(args.file2)
    for ix,(ph1, ph2) in enumerate(zip(tg1.get_phoneme(), tg2.get_phoneme())):
        if ph1.text[:2] == 'sp':
            if abs(ph1.xmin - ph2.xmin) > 0.0001:
                Print_(fnb, tg1, ix, ix+1)


def main(args):
    if args.ph:
        diff_phoneme(args)
    if args.sp_time:
        diff_sp_time(args)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
    # parser.add_argument('-s', '--opt_str', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
    parser.add_argument('--ph', action='store_true')
    parser.add_argument('--sp_time', action='store_true')
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
