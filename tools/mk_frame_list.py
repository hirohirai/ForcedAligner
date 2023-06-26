#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/09/13

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np
import numpy.random

from utils.TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)



def main_rand(args):

    if args.width > 1:
        logging.error(f"{args.width} Not SUPPORTED")

    for ll in args.input:
        fid = ll.strip().split(',')
        #tgfn = f'{args.tg_dir}/{fid[0]}/{fid[1]}.TextGrid'
        tgfn = f'{args.tgt_dir}/{fid[0]}/{fid[1]}.npz'
        tgt = np.load(tgfn)
        tgt = tgt['target']
        num = len(tgt)
        nn = num - args.st - args.ed
        nn2 = int(nn / args.num) * args.num
        st = args.st + int((nn - nn2)/2)
        ixlst = np.arange(st,st+nn2)
        numpy.random.shuffle(ixlst)
        for ix in range(0, nn2, args.num):
            ixs = ixlst[ix:ix+args.num]
            ixs.sort()
            print(f'{fid[0]},{fid[1]}:{ixs[0]}', end='')
            for nn in ixs[1:]:
                print(f';{nn}', end='')
            print("")


def main(args):
    for ll in args.input:
        fid = ll.strip().split(',')
        tgfn = f'{args.tgt_dir}/{fid[0]}/{fid[1]}.npz'
        tgt = np.load(tgfn)
        tgt = tgt['target']
        num = len(tgt)

        if args.width < 1:
            for nn in range(args.st, num - args.ed):
                print(f'{fid[0]},{fid[1]}:{nn}')
        else:
            for nn in range(args.st, num - args.ed, args.width - args.margin):
                ed = nn+args.width
                if ed > num - args.ed:
                    ed = num - args.ed
                if ed - nn > 1:
                    print(f'{fid[0]},{fid[1]}:{nn}:{ed}')



if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
    parser.add_argument('-t', '--tgt_dir', default='data/targets')
    parser.add_argument('--st', type=int, default=2)
    parser.add_argument('--ed', type=int, default=2)
    parser.add_argument('-n', '--num', type=int, default=8)
    parser.add_argument('-w', '--width', type=int, default=-1)
    parser.add_argument('-m', '--margin', type=int, default=0, help='重なる個数')
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

    if args.width > 0:
        main(args)
    else:
        main_rand(args)
