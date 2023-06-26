#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/05/09

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import utils.Tts
import utils.TextGrid
import utils.text.kana

# ログの設定
logger = logging.getLogger(__name__)


def main(args):
    nn = args.st
    for ll in args.input:
        txt = ll.strip()
        kana = args.input.readline().strip()
        tts = utils.Tts.Tts()
        tts.text = txt
        ks = utils.text.kana.KanaSent()
        ks.set_kana(kana)
        tts.from_kanaSent(ks)
        tg =utils.Tts.tts_to_textGrid(tts)
        fn = f'{args.fnb}{nn:02}.TextGrid'
        with open(fn, 'w') as ofs:
            print(tg, file=ofs)
        nn += 1


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
    # parser.add_argument('-s', '--opt_str', default='')
    parser.add_argument('--fnb', default='atr503/TextGrid/K')
    parser.add_argument('--st', type=int, default=1)
    parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
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
