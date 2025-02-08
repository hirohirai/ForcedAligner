#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/08/09

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import wave

from utils.TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)


class RMPos:
    def __init__(self, ee, wv_max=-1.0):
        self.st = ee[0]
        self.ed = ee[1]
        self.fnum = ee[2]
        self.wv_max = wv_max

    def get_st(self):
        return float(self.st)

    def get_ed(self):
        if self.ed == 'None':
            return self.wv_max
        return float(self.ed)

class RtMriPos:
    def __init__(self, fn, wv_max):
        self.body = {}
        for ll in open(fn):
            ee = ll.strip().split()
            self.body[ee[0]] = RMPos(ee[1:], wv_max)

    def __getitem__(self, item):
        return self.body[item]




def mk_TG(tgfn, wavfn, ttime, otime, frlbl, otgfn):
    tg = TextGrid(tgfn)
    wr = wave.open(wavfn,'rb')
    fr = wr.getframerate()
    fn = wr.getnframes()
    wvlen = float(fn) / fr
    tg.add_time(otime-ttime)
    tg.xmin = 0.0
    tg.xmax = wvlen
    for itm in tg.item:
        itm.xmin = 0.0
        itm.xmax = wvlen
    tg.addStEd()
    tg.addFrameNum(ttime, fn=frlbl, frate=0.0368)
    tg.correct_times()
    with open(otgfn, 'w') as ofs:
        print(tg, file=ofs)


def ck_data_err(trpd, mrpd):
    if trpd.get_ed() < mrpd.get_st() or trpd.get_st() > mrpd.get_ed():
        return True
    stp = trpd.get_st() if trpd.get_st() > mrpd.get_st() else mrpd.get_st()
    edp = trpd.get_ed() if trpd.get_ed() < mrpd.get_ed() else mrpd.get_ed()

    if (edp - stp) / (mrpd.get_ed() - mrpd.get_st()) < 0.5:
        return True

    return False

def main(args):
    mrp = RtMriPos(args.myfile, args.wv_max)
    trp = RtMriPos(args.rtfile, args.wv_max)

    for fnum in open(args.file):
        fnum = fnum.strip()
        tgfn = f'{args.idir}/{fnum.upper()}.TextGrid'
        wavfn = f'{args.wvdir}/{fnum}.WAV'
        ttime = trp[fnum].get_st()
        otime = mrp[fnum].get_st()
        frlbl = f'{args.lbl}{trp[fnum].fnum}'
        otgfn = f'{args.odir}/{fnum}.TextGrid'

        if ck_data_err(trp[fnum], mrp[fnum]):
            logging.error(f'data okasii {fnum}')

        mk_TG(tgfn, wavfn, ttime, otime, frlbl, otgfn)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-l', '--lbl', default='a')
    parser.add_argument('--myfile', default='')
    parser.add_argument('--rtfile', default='')
    parser.add_argument('--idir', default='/home/hirai/work_local/Speech/DBS_/rtmri-atr503/TextGrid/s1')
    parser.add_argument('--odir', default='.')
    parser.add_argument('--wvdir',
                        default='/home/hirai/Dropbox/realTimeMRI/ATR503/Maekawa/WAV/ATR503_sentences_selected')
    parser.add_argument('--wv_max', type=float, default=18.88)
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
