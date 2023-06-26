#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/08/22

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np

# ログの設定
logger = logging.getLogger(__name__)

F0_LZERO = -5
CAP_LZERO = 5

F0_ZERO = 30
CAP_ZERO = 0

class WorldCof:
    def __init__(self, fn=None, mgcP=2, capP=45):
        if fn:
            self.load(fn)
        elif capP>0 and mgcP>0:
            self.mgcP = mgcP
            self.capP = capP
            self.av = np.zeros(self.capP + self.mgcP + 1)
            self.std = np.ones(self.capP+self.mgcP+1)

    def clear(self):
        self.xsum = np.zeros(self.capP + self.mgcP + 1)
        self.xxsum = np.zeros(self.capP + self.mgcP + 1)
        self.f0n = 0
        self.mgcn = 0
        self.capn = np.zeros(self.capP)

    def set_dat(self, f0, mgc, cap):
        lf0 = np.log(f0[f0>0])
        self.xsum[0] += np.sum(lf0)
        self.xxsum[0] += np.sum(lf0*lf0)
        self.f0n += len(lf0)
        self.xsum[1:1 + self.mgcP] += np.sum(mgc, axis=0)
        self.xxsum[1:1 + self.mgcP] += np.sum(mgc*mgc, axis=0)
        self.mgcn += mgc.shape[0]
        for ii in range(self.capP):
            cap_ = cap[cap[:,ii]<0.0, ii]
            self.xsum[1 + self.mgcP + ii] += np.sum(cap_)
            self.xxsum[1 + self.mgcP + ii] += np.sum(cap_*cap_)
            self.capn[ii] += len(cap_)

    def add(self, wc2):
        self.xsum += wc2.xsum
        self.xxsum += wc2.xxsum
        self.f0n += wc2.f0n
        self.mgcn += wc2.mgcn
        self.capn += wc2.capn

    def calc_stat(self, save_flg=False):
        nn = np.hstack([self.f0n, np.ones(self.mgcP)*self.mgcn, self.capn])
        self.av = self.xsum / nn
        self.std = np.sqrt(self.xxsum / nn - self.av* self.av)
        if not save_flg:
            del self.xsum
            del self.xxsum
            del self.f0n
            del self.mgcn
            del self.capn


    def save(self, fn):
        np.savez(fn, self.av, self.std, self.mgcP, self.capP)

    def load(self, fn):
        buf = np.load(fn)
        self.av = buf['arr_0']
        self.std = buf['arr_1']
        self.mgcP = int(buf['arr_2'])
        self.capP = int(buf['arr_3'])

    def encode(self,indata):
        f0 = indata[:,0]
        mgc = indata[:,1:self.mgcP+1]
        cap = indata[:,-self.capP:]
        return self.encode0(f0, mgc, cap)

    def encode0(self, f0_, mgc_, cap_):
        f0_ix = f0_> 0
        f0 = np.ones(f0_.shape) * F0_LZERO
        f0[f0_ix] = (np.log(f0_[f0_ix]) - self.av[0]) / self.std[0]
        f0 = f0.reshape(len(f0), 1)
        mgcP = mgc_.shape[1]

        mgc = (mgc_ - self.av[1:mgcP+1]) / self.std[1:mgcP+1]

        cap_ix = cap_ == 0
        cap = (cap_ - self.av[-self.capP:]) / self.std[-self.capP:]
        cap[cap_ix] = CAP_LZERO
        return np.hstack([f0, mgc, cap])


    def decode(self, x_in):
        f0 = x_in[:, 0] * self.std[0] + self.av[0]
        f0[f0 < F0_ZERO] = 0.0
        f0 = f0.reshape(len(f0), 1)

        mgc = (x_in[:, 1:-self.capP] * self.std[1:-self.capP]) + self.av[1:-self.capP]

        cap = (x_in[:, -self.capP:] * self.std[-self.capP:]) + self.av[-self.capP:]
        cap[cap>=CAP_ZERO] = 0.0

        return np.hstack([f0, mgc, cap])


class StatsCof:
    def __init__(self, fn=None, param_n=0):
        if fn:
            self.load(fn)
        elif param_n>0:
            self.param_n = param_n
            self.av = np.zeros(self.param_n)
            self.std = np.ones(self.param_n)

    def clear(self):
        self.xsum = np.zeros(self.param_n)
        self.xxsum = np.zeros(self.param_n)
        self.num = 0

    def set_dat(self, stft):
        self.xsum += np.sum(stft, axis=0)
        self.xxsum += np.sum(stft*stft, axis=0)
        self.num += len(stft)

    def add(self, wc2):
        self.xsum += wc2.xsum
        self.xxsum += wc2.xxsum
        self.num += wc2.num

    def calc_stat(self, save_flg=False):
        if self.num == 0:
            logging.error("ERROR DIV ZERO")
        self.av = self.xsum / self.num
        if np.any(np.isnan(self.av)):
            logging.error("ERROR NAN")
        self.std = np.sqrt(self.xxsum / self.num - self.av * self.av)
        if not save_flg:
            del self.xsum
            del self.xxsum

    def save(self, fn):
        np.savez(fn, self.av, self.std, self.num)

    def load(self, fn):
        buf = np.load(fn)
        self.av = buf['arr_0']
        self.std = buf['arr_1']
        self.num = int(buf['arr_2'])
        self.param_n = len(self.av)

    def encode(self, stft):
        return (stft - self.av) / self.std

    def decode(self, stft):
        return stft * self.std + self.av



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
