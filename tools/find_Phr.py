#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2025/10/31

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
# from utils.Tts import textGrid_to_Tts
from utils.TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)


def strip_sword(txt):
    if txt.endswith('AYOR'):
        txt = txt[:-5]
    return txt.split(':')[-1]

def find_phrase(tgfn):
    tg = TextGrid(tgfn)
    fnb = tg.get_text().strip().split()[0]

    kana = tg.get_item('kana')
    sword = tg.get_item('word')
    for ix,ka in enumerate(kana.intervals):
        if ka.text[0] == '，':
            six = sword.find_st(ka.xmin)
            usiro = strip_sword(sword.intervals[six].text)
            if sword.intervals[six-1].text.startswith('sp'):
                mix = six - 2
            else:
                mix = six - 1
            mae = strip_sword(sword.intervals[mix].text)

            if len(mae)==1 and mix>0:
                mae = strip_sword(sword.intervals[mix-1].text) + mae

            print(f'{fnb} {mae} ， {usiro} "{ka.text}"')




def main(args):
    find_phrase(args.file)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
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
