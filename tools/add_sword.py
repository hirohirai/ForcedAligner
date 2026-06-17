#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/06/06

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
from utils.TextGrid import TextGrid
import utils.Tts as Tts

# ログの設定
logger = logging.getLogger(__name__)


def add_textgrid(tg, swords):
    tg.correct_word()
    tg.clear_swords()
    ks = tg.get_kanaSent()
    if isinstance(swords, list):
        ks.add_sword(swords)
    else:
        ks.add_sword(swords.strip().split())
    tts = Tts.Tts()
    tts.from_kanaSent(ks)
    tts.set_time(tg, True)
    tts.xmin = tg.xmin
    tts.xmax = tg.xmax
    tts.text = tg.get_text()
    tgo = Tts.tts_to_textGrid(tts)
    tgo.copyFrameNum(tg)
    tgo.addStEd()

    tgo.correct_times()

    return tgo

def get_swords(fn):
    with open(fn, 'r', encoding='utf-8') as ifs:
        swords = []
        for ll in ifs:
            swords.append(ll.strip())
        return swords

def main2(args):
    swords = get_swords(args.sword)
    tg = TextGrid(args.ifn)
    otg = add_textgrid(tg, swords)

    with open(args.ofn, 'w', encoding='utf-8') as ofs:
        print(str(otg), file=ofs)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ifn', default='./')
    parser.add_argument('-o', '--ofn', default='./out/')
    parser.add_argument('--sword', default='./a.txt')
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

    main2(args)
