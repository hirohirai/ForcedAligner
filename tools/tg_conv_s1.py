#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    s1が２種類あるので、千葉工大用のファイルセットをRTMRIへ変換するためのリストを作成
    Author: hirai
    Data: 2025/02/07

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import scipy.io.wavfile
import numpy as np

from utils.TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)

FR_RATE=27.1739

def get_wav_len(fn):
    sr, data = scipy.io.wavfile.read(fn)
    return len(data) / sr

class SplitLst:
    def __init__(self, fn, mrinum, st, ed):
        self.fn = fn
        self.mrinum = int(mrinum)
        self.st = float(st)
        self.stnum = int(np.round(self.st/FR_RATE))
        self.wlen = None
        if ed == 'None':
            self.ed = None
        else:
            self.ed = float(ed)

    def setEdTime(self, wvdir):
        wvfn = f'{wvdir}/{self.fn}.WAV'
        if os.path.isfile(wvfn):
            self.wlen = get_wav_len(wvfn)
            if self.ed is None:
                self.ed = self.st + self.wlen
        else:
            self.wlen = None


def get_st_time_info(frm):
    buf1, buf2 = frm.text.split(':')
    tg_date_ab = buf1[0]
    tg_ss_num = int(buf1[1:])
    tg_fr_num = int(buf2)
    tg_st_time = tg_fr_num * (1/FR_RATE) - (frm.xmax-frm.xmin)
    return tg_st_time, tg_date_ab, tg_ss_num


def chk_diff_time(spl, st_time, len_time, ss_num):
    ed_time = st_time + len_time
    if spl.mrinum != ss_num:
        return None
    st = spl.st if spl.st> st_time else st_time
    ed = spl.ed if spl.ed < ed_time else ed_time
    dur = ed - st
    if dur < (ed_time - st_time) * 0.8:
        return None

    return st_time - spl.st



def main(args):
    for ll in open(args.rtfile):
        ee = ll.strip().split()
        spl = SplitLst(ee[0], ee[3], ee[1], ee[2])
        if '_' in spl.fn:
            continue
        spl.setEdTime(args.wv_dir)
        if spl.wlen is None:
            print(f"NoWav {spl.fn}")
            continue
        tgFn = f'{args.tg_dir}/{spl.fn.upper()}.TextGrid'
        tg = TextGrid(tgFn)
        st_time, date_ab, ss_num = get_st_time_info(tg.get_frame(0))
        if args.date_ab == date_ab:
            diff_time = chk_diff_time(spl, st_time, tg.xmax, ss_num)
        else:
            diff_time = None



        if diff_time is None:
            print(f"None {spl.fn}")
        else:
            st, ed = tg.getStEd()
            if diff_time + st < 0.0:
                print(f"St {spl.fn}")
            if diff_time + ed > spl.wlen:
                print(f"Ed {spl.fn}")
            tg.add_time(diff_time)
            tg.set_xmax_xmin(spl.wlen)
            tg.addFrameNum(spl.st, fn=f'{date_ab}{ss_num}')
            ofn = f'{args.odir}/{spl.fn}.TextGrid'
            with open(ofn, 'w') as ofs:
                print(tg, file=ofs)



if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('rtfile')
    parser.add_argument('--wv_dir', default='wav')
    parser.add_argument('--tg_dir', default='TextGrid')
    parser.add_argument('--odir', default='out_open_time')
    parser.add_argument('--date_ab', default='a')
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
