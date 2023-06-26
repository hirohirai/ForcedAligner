#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/03/18

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import pickle
import numpy as np

# ログの設定
logger = logging.getLogger(__name__)


class DurLogLikelihood:
    def __init__(self, durstats=None, fn=None):
        self.av = {}
        self.std = {}
        if durstats:
            self.set_data(durstats)
        if fn:
            self.load(fn)

    def set_data(self, durstats):
        for ky in durstats.keys():
            av_, std_ = durstats.get(ky)
            self.av[ky] = av_
            self.std[ky] = std_

    def get_LogLikelihood(self, lbl, dur):
        if lbl.startswith('sp'):
            lbl = 'sp'
            return 0.0
        elif lbl.startswith("#"):
            lbl = 'sp'
            return 0.0
        elif lbl.endswith("H"):
            lbl = lbl[0]
        return -np.log(2*np.pi) - np.log(self.std[lbl]) - (dur-self.av[lbl])**2 / self.std[lbl]**2 / 2

    def save(self, fn):
        with open(fn, 'wb') as ofs:
            pickle.dump([self.av, self.std], ofs)

    def load(self, fn):
        with open(fn, 'rb') as ifs:
            buf = pickle.load(ifs)
            self.av = buf[0]
            self.std = buf[1]

class DurStats:
    def __init__(self):
        self.sum = {}
        self.sum2 = {}
        self.num = {}

    def add(self, lbl, dur):
        if lbl in self.sum:
            self.sum[lbl] += dur
            self.sum2[lbl] += (dur*dur)
            self.num[lbl] += 1
        else:
            self.sum[lbl] = dur
            self.sum2[lbl] = dur * dur
            self.num[lbl] = 1

    def merge(self, dic2):
        for ky in dic2.sum.keys():
            if ky in self.sum:
                self.sum[ky] += dic2.sum[ky]
                self.sum2[ky] += dic2.sum2[ky]
                self.num[ky] += dic2.num[ky]
            else:
                self.sum[ky] = dic2.sum[ky]
                self.sum2[ky] = dic2.sum2[ky]
                self.num[ky] = dic2.num[ky]

    def keys(self):
        return self.sum.keys()

    def get(self, ky):
        if ky in self.sum:
            av = self.sum[ky] / self.num[ky]
            std = np.sqrt(self.sum2[ky] / self.num[ky] - av * av)
            return av, std
        else:
            None

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
