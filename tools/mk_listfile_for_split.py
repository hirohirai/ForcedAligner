#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/06/07

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import os.path
import glob

from utils.TextGrid import TextGrid


# ログの設定
logger = logging.getLogger(__name__)


def mk_list(tgfn, fr_fps, stw, edw):
    fnb = os.path.splitext(os.path.basename(tgfn))[0]
    frw = 1/ fr_fps
    tg = TextGrid(tgfn)
    if len(tg.item) != 1:
        logger.warning(f'{tgfn}: item > 1')
    lst = []
    for itm in tg.item:
        if itm.name == '"かな"' or itm.name == '"kana"':
            for itv in itm.intervals:
                ee = itv.text.strip().split(':')
                if len(ee[0]) > 2:
                    lst.append((ee[0], itv.xmin, itv.xmax))
    olst = []
    if len(lst) == 0:
        logger.error(f'{tgfn}: fmt error')
    else:
        for ix, ll in enumerate(lst):
            st = round((ll[1] - stw) / frw) * frw
            ed = round((ll[2] + edw) / frw) * frw
            if ix > 0 and st < lst[ix-1][2]:
                logger.warning(f'{tgfn}: {ix} {st} < last end')
            if ix < len(lst)-1 and lst[ix+1][1] < ed:
                logger.warning(f'{tgfn}: {ix} {ed} > next start')
            olst.append([ll[0], ll[1], ll[2], st, ed, int(fnb)])

    return olst

def print_all(olst):
    for elm_ in olst:
        elm = [str(ee) for ee in elm_]
        out = ' '.join(elm)
        print(out)


def rename_fn(olst):
    fnNum = {}
    for ix in range(len(olst)-1, -1, -1):
        if olst[ix][0] in fnNum:
            fnNum[olst[ix][0]] += 1
            olst[ix][0] += f'_{fnNum[olst[ix][0]]}'
        else:
            fnNum[olst[ix][0]] = 1

    return olst

def main(args):
    olst_all = []
    for fn in glob.glob(args.file):
        olst_all += mk_list(fn, args.fps, args.stw, args.edw)

    olst_all.sort(key=lambda x:x[5])

    olst_all = rename_fn(olst_all)

    print_all(olst_all)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    # parser.add_argument('-s', '--opt_str', default='')
    parser.add_argument('--fps', type=float, default=27.1739)
    parser.add_argument('--stw', type=float, default=0.100)
    parser.add_argument('--edw', type=float, default=0.200)
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
