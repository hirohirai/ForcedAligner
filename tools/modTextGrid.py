#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2026/04/24

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
# ログの設定
logger = logging.getLogger(__name__)

from utils.TextGrid import TextGrid

def chgFrameLbl(tg):
    for fr in tg.get_frame():
        #abc = fr.text[0]
        #stu = ord(abc) - ord('a') + ord('s')
        #fr.text = chr(stu) + fr.text[1:]
        fr.text = 's' + fr.text[1:]
    return tg


def main(args):
    tg = TextGrid(args.file)
    tg = chgFrameLbl(tg)
    print(tg)

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