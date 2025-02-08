#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    split_file.py
    Author: hirai
    Data: 2022/03/1
"""
import sys, os
import os.path
import librosa
import scipy.io.wavfile
import numpy as np
import argparse
import logging
import jaconv
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from utils.TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)


def wavread(fn, fs=-1.0) :
    if fs<0:
        fs, wv = scipy.io.wavfile.read(fn)
    else:
        (wv, fs) = librosa.load(fn, sr=fs)
        wv = np.round(wv * 32768)
        ix = wv>32767
        wv[ix] = 32767
        ix = wv<-32768
        wv[ix] = -32768
        wv = np.asarray(wv,np.int16)
    return fs, wv

def wavwrite(fn, wv, fs):
    scipy.io.wavfile.write(fn, fs, wv)

def kana_to_julkana(kana):
    kana = kana.replace('＿',' sp').replace('／','').replace('’','').replace('。','').replace('｜','').replace('sp', ' sp').replace('_', ' sp').replace('  ', ' ').replace('．','')
    return jaconv.kata2hira(kana)

def save_julius_files(wav, fs, stp, edp, kana, yomi, fnb):
    wavwrite(fnb+'.wav', wav[stp:edp], fs)
    with open(fnb+'.txt', 'w') as f:
        print(yomi, file=f)
    with open(fnb+'.kana', 'w') as f:
        print(kana, file=f)


def split_file_for_julius_seg(input_wav, input_tgrid, odir, fs, cflg):
    fs, wav = wavread(input_wav, fs)
    tg = TextGrid(input_tgrid)
    bname = os.path.splitext(os.path.basename(input_wav))[0]

    ix = 0
    for wd in tg.item[0].intervals:
        if len(wd.text)>0 and wd.text.strip() != 'sp':
            #fnb = f'{odir}/{bname}_{ix:02}'
            #yomi = kana_to_julkana(wd.text)
            if wd.text.find(':') < 0:
                continue
            fnb, yomi = wd.text.split(':')
            yomi = kana_to_julkana(yomi)
            yomi = yomi.replace('　', ' ').strip()
            ofnb = f'{odir}/{fnb}'

            sttime = wd.xmin - 0.100
            edtime = wd.xmax + 0.200
            stp = round(sttime * fs)
            edp = round(edtime * fs)
            logger.debug(stp, edp, wd.text, yomi, ofnb)
            print(f'{fnb} {bname} {sttime} {edtime}')
            if cflg:
                if edp - stp <= 0.0:
                    logger.error(f'TextGrid Time Error')
                if len(wd.text) ==0 or len(yomi) == 0:
                    logger.error(f'TextGrid text Error')
            else:
                save_julius_files(wav, fs, stp, edp, wd.text, yomi, ofnb)
            ix += 1


def set_init_time_elem(elem, ph_st, ph_ed, wav_len):
    if elem[0].text == '#':
        elem[0].xmin = 0.0
        elem[0].xmax = ph_st
        stp = 1
    else:
        stp = 0
    if elem[-1].text == '#':
        edp = len(elem)-1
    else:
        edp = len(elem)

    width = (ph_ed - ph_st) / (edp - stp)
    posi = ph_st
    for ii in range(stp, edp):
        elem[ii].xmin = posi
        posi += width
        elem[ii].xmax = posi

    if elem[-1].text == '#':
        elem[-1].xmin = posi
        elem[-1].xmax = wav_len


def set_init_time(tg, ph_st, ph_ed, wav_len, wav_st, fnb, fps):
    tg.xmax = wav_len
    snt = tg.get_sent()
    if snt:
        if len(snt) == 1:
            snt[0].xmin = ph_st
            snt[0].xmax = ph_ed
        elif len(snt) == 3:
            snt[0].xmin = 0.0
            snt[0].xmax = ph_st
            snt[1].xmin = ph_st
            snt[1].xmax = ph_ed
            snt[2].xmin = ph_ed
            snt[2].xmax = wav_len
        else:
            print("sent has No TEXT")
    else:
        print("TextGrid has No sent")

    phn = tg.get_phoneme()
    if phn and len(phn) > 0:
        set_init_time_elem(phn, ph_st, ph_ed, wav_len)
    else:
        print("TextGrid has No phoneme")

    wrd = tg.get_word()
    if wrd and len(wrd) > 0:
        set_init_time_elem(wrd, ph_st, ph_ed, wav_len)
    else:
        print("TextGrid has No word")

    swrd = tg.get_sword()
    if swrd and len(swrd) > 0:
        set_init_time_elem(swrd, ph_st, ph_ed, wav_len)
    else:
        print("TextGrid has No sword")

    tg.addFrameNum(wav_st, fps, fnb)

    tg.addStEd()
    return tg

def split_file_for_textGrid(fnb, st, ed, ph_st, ph_ed, iwavfn, owavfn, itgfn, otgfn, fps=27.1739):
    fs, wav = wavread(iwavfn)
    stp = round(fs * st)
    edp = round(fs * ed)
    wavwrite(owavfn, wav[stp:edp], fs)

    tg = TextGrid(itgfn)
    tg = set_init_time(tg, ph_st, ph_ed, ed - st, st, fnb, fps)
    with open(otgfn, 'w') as ofs:
        print(tg, file=ofs)




def split_file_for_textGrid_all(lstfn, iWavDir, iTgDir, oWavDir, oTgDir, mrifnb_, wvext, tg_lower=False, endtime=18.1784):
    with open(lstfn) as ifs:
        for ll in ifs:
            ee = ll.strip().split()
            fnb = ee[0]
            #st = float(ee[3])
            st = float(ee[1])
            #ph_st = float(ee[1]) - st
            ph_st = 0.1
            #ed = float(ee[4])
            if ee[2] == 'None':
                ed = endtime
            else:
                ed = float(ee[2])
            #ph_ed = float(ee[2]) - st
            ph_ed = ed - st - 0.1
            fnb_ = fnb.split('_')[0]
            mrifnb = f'{mrifnb_}{ee[-1]}'
            iwavfn = f'{iWavDir}/{ee[-1]}{wvext}'
            owavfn = f'{oWavDir}/{fnb}.wav'
            if tg_lower:
                itgfn = f'{iTgDir}/{fnb_}.TextGrid'
            else:
                itgfn = f'{iTgDir}/{fnb_.upper()}.TextGrid'
            otgfn = f'{oTgDir}/{fnb}.TextGrid'
            #print(f"write {owavfn} {otgfn}")
            split_file_for_textGrid(mrifnb, st, ed, ph_st, ph_ed, iwavfn, owavfn, itgfn, otgfn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wav")
    parser.add_argument("input_tgrid")
    parser.add_argument("--fnb", default="")
    parser.add_argument('-l', "--list", default="")
    parser.add_argument("-m", "--mode", default="TG")
    parser.add_argument("--Fs", type=int, default=-1)
    parser.add_argument("--owav", default="out")
    parser.add_argument('-o', "--odir", default="out")
    parser.add_argument('--wvext', default='.WAV')
    parser.add_argument('--edtime', type=float, default=18.8784)
    parser.add_argument('--tglower', action='store_true')
    parser.add_argument('--check', '-c', action='store_true')
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

    if args.mode == 'TG':
        split_file_for_textGrid_all(args.list, args.input_wav, args.input_tgrid, args.owav, args.odir, args.fnb, args.wvext, args.tglower, args.edtime)
    else:
        split_file_for_julius_seg(args.input_wav, args.input_tgrid, args.odir, args.Fs, args.check)

