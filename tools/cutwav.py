#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/03/26

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import librosa
import scipy.io.wavfile as wavfile
import numpy as np

from utils.TextGrid import TextGrid
# ログの設定
logger = logging.getLogger(__name__)


def cut_main(args):
    tg = TextGrid(args.tgfile)
    frms  = tg.get_frame(0).text.split(':')
    if frms[0][0] == 'a':
        wvdir = args.wavdira
    else:
        wvdir = args.wavdirb
    wvfname = f'{wvdir}/{frms[0][1:]}.WAV'
    wv, sr = librosa.load(wvfname, sr=20000)
    fps = 1 / args.mri_rate
    wv_st = round((int(frms[1])-1) * fps * sr)
    wv_len = round(tg.xmax * sr)
    wv = wv[wv_st:wv_st+wv_len+1] * 32767 * args.cof
    wavfile.write(args.ofile, sr, wv.astype(np.int16))

def noise_cut_main(args):
    wv,sr = librosa.load(args.ifile, sr=20000)
    #noi = wv[2048:6144] / 2.3 * 32767 # maekawa
    noi = wv[2048:6144] * args.cof * 32767
    wavfile.write(args.ofile, sr, noi.astype(np.int16))


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('tgfile')
    parser.add_argument('ofile')
    parser.add_argument('-c', '--cof', type=float, default=1.0)
    parser.add_argument('--mri_rate', type=float, default=27.17)
    parser.add_argument('--wavdira', default='/home/hirai/Dropbox/realTimeMRI/20220221/s1/WAV_R/1001/denoised_frcrn')
    parser.add_argument('--wavdirb', default='/home/hirai/Dropbox/realTimeMRI/20220222/s1/WAV_R/1001/denoised_frcrn')
    parser.add_argument('--mode', default='')
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

    if args.mode == 'noise':
        noise_cut_main(args)
    else:
        cut_main(args)
