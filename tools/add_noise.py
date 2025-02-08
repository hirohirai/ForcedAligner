#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/06/23

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np

# ログの設定
logger = logging.getLogger(__name__)

def index_zero(wv):
    ix = wv == 0
    ix2 = ix[1:]
    ix3 = ix[2:]
    ix[1:-1] = ix[:-2] & ix2[:-1] & ix3
    ix[0] = ix[-1] = False
    return ix

def add_noise(wv):
    ix = index_zero(wv)
    if wv.dtype == np.int16:
        wv[ix] = np.array(np.random.rand(len(wv[ix]))*6 -3, dtype=wv.dtype)
    else:
        max_ = np.max(np.abs(wv))
        wv += np.array(np.random.randn(len(wv)), dtype=wv.dtype) * max_ / 32767
    return wv



def main(args):
    pass


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
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
