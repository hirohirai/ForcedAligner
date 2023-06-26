#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/05/19

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging


from utils.TextGrid import TextGrid
from utils.Tts import textGrid_to_Tts, tts_to_textGrid

# ログの設定
logger = logging.getLogger(__name__)


def main(args):
    tg1 = TextGrid(args.file1)
    tts1 = textGrid_to_Tts(tg1)
    tts1.div_all()
    tg1o = tts_to_textGrid(tts1)
    tg2 = TextGrid(args.file2)
    tts2 = textGrid_to_Tts(tg2)
    tts2.div_all()
    tg2o = tts_to_textGrid(tts2)

    err=0.0
    max_err=0.0
    max_ix = -1
    for ix, (ph1, ph2) in enumerate(zip(tg1o.get_phoneme(), tg2o.get_phoneme())):
        if ph1.text != ph2.text:
            logging.error(f'NQ {ph1.text} {ph2.text} Ix={ix}')
        ee = abs(ph1.xmax - ph2.xmax)
        if ee > max_err:
            max_ix = ix
            max_err = ee
        err += ee

    print(err / (ix+1), max_err, max_ix)



if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
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
