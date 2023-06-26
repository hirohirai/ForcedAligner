#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/06/24

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

import os.path
import numpy as np
import librosa
import scipy

import pyworld as pw

# ログの設定
logger = logging.getLogger(__name__)


def savewav(fn, wv, rate=20000) :
    if type(wv[0]) != np.int16 :
        if np.max(wv) < 1.0:
            wv *= 32767
        wv = np.round(wv)
        ix = wv>32767
        wv[ix] = 32767
        ix = wv<-32768
        wv[ix] = -32768
        wv = np.asarray(wv,np.int16)
    scipy.io.wavfile.write(fn, rate, wv)

def do_world_f0(wv, sr, _f0):
    t = np.arange(len(_f0)) * 0.005
    f0 = pw.stonemask(wv, _f0, t, sr)  # pitch refinement
    sp = pw.cheaptrick(wv, f0, t, sr)  # extract smoothed spectrogram
    ap = pw.d4c(wv, f0, t, sr)
    return sp, ap, f0


def do_world(wv, sr):
    f0, sp, ap = pw.wav2world(wv, sr)
    return sp, ap, f0


def code_world(spc, ap, sr, order):
    mgc = pw.code_spectral_envelope(spc, sr, order)
    cap = pw.code_aperiodicity(ap, sr)
    return mgc, cap


def ana(wvfn, f0fn, sr, ofn, mgc_order):
    wv, sr = librosa.load(wvfn, sr=sr, dtype=np.float64)
    if f0fn:
        f0 = np.fromfile(f0fn, dtype=np.float64)
        spc, ap, f0 = do_world_f0(wv, sr, f0)
    else:
        spc, ap, f0 = do_world(wv, sr)

    if mgc_order<=0:
        np.savez(ofn, f0=f0, spc=spc, ap=ap)
    else:
        mgc, cap = code_world(spc, ap, sr, mgc_order)
        np.savez(ofn, f0=f0, mgc=mgc, cap=cap)


def syn(ifn, ofn, sr):
    para = np.load(ifn)
    if 'mgc' in para.files:
        fft_sz = pw.get_cheaptrick_fft_size(sr)
        sp_ = pw.decode_spectral_envelope(para['mgc'], sr, fft_sz)
        ap_ = pw.decode_aperiodicity(para['cap'], sr, fft_sz)
    else:
        sp_ = para['spc']
        ap_ = para['ap']
    syn = pw.synthesize(para['f0'], sp_, ap_, sr)
    savewav(ofn, syn, sr)


def main(args):
    if args.ofn:
        if args.ofn[0] in './':
            ofn = args.ofn
        else:
            ofn = f'{args.odir}/{args.ofn}'
    else:
        fname = os.path.splitext(os.path.basename(args.file))[0]
        ofn = f'{args.odir}/{fname}'
    if args.syn:
        if not ofn.endswith('.wav'):
            ofn += '.wav'
        syn(args.file, ofn, args.Fs)
    else:
        ana(args.file, args.f0, args.Fs, ofn, args.order)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--f0', default='')
    parser.add_argument('--odir', default='./')
    parser.add_argument('-o', '--ofn', default='')
    parser.add_argument('--order', type=int, default=45)
    parser.add_argument('--Fs', type=int, default=20000)
    parser.add_argument('--syn', '-s', action='store_true')
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
