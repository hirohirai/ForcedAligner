#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2025/02/07

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import scipy.io.wavfile

# ログの設定
logger = logging.getLogger(__name__)

class SplitLst:
    def __init__(self, fn, mrinum, st, ed):
        self.fn = fn
        self.mrinum = int(mrinum)
        self.st = float(st)
        if ed == 'None':
            self.ed = None
        else:
            self.ed = float(ed)


def get_wav_len(fn):
    sr, data= scipy.io.wavfile.read(fn)
    return len(data)/sr

def main(args):
    mylst = []
    rtlst = []
    for ll in open(args.myfile):
        ee = ll.strip().split()
        mylst.append(SplitLst(ee[0], ee[1], ee[2], ee[3]))
    for ll in open(args.myfile):
        ee = ll.strip().split()
        rtlst.append(SplitLst(ee[0], ee[1], ee[2], ee[3]))

    for sl in rtlst:
        if sl.ed is None:
            wvfn = f'{args.wavdir}/{sl.fn}.WAV'
            wlen = get_wav_len(wvfn)
            sl.ed = sl.st + wlen





if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('myfile')
    parser.add_argument('rtfile')
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
