#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2025/08/19

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import copy

from utils.TextGrid import TextGrid, Interval
from utils.Tts import textGrid_to_Tts, tts_to_textGrid
from utils.text.rtMRI import MORA_VOW
from utils.text.rtMRI2kana import rom2kana_c

# ログの設定
logger = logging.getLogger(__name__)


def mora_tier(tg):
    lcons = ''
    ltim = 0.0
    mr_tier = []
    for ph in tg.get_phoneme():
        if ph.text == '#':
            mr_tier.append(Interval(ph.text, ltim, ph.xmax))
            ltim = ph.xmax
        if ph.text.startswith('sp'):
            mr_tier.append(Interval(ph.text, ltim, ph.xmax))
            ltim = ph.xmax
        elif ph.text in MORA_VOW:
            ka = rom2kana_c(lcons, ph.text)
            mr_tier.append(Interval(ka, ltim, ph.xmax))
            ltim = ph.xmax
            lcons=''
        else:
            lcons = ph.text

    return mr_tier


def delete_ayor(tg):
    tgo = copy.copy(tg)
    return tgo


def addMora(tg):
    tts = textGrid_to_Tts(tg)
    tts.div_all()
    for wrd in tts.words:
        for ix, mr in enumerate(wrd.moras):
            if mr.hasCL == True:
                mr.clen = mr.clen + mr.CL_len
                mr.CL_len = 0.0
                mr.hasCL = False
            if mr.hasJ == True:
                mr.clen = mr.clen + mr.J_len
                mr.J_len = 0.0
                mr.hasJ = False

    tgo = tts_to_textGrid(tts)
    mora = mora_tier(tgo)
    tg.addMora(mora)
    return tg

def addAYOR(tg):
    tg.addAYOR()
    ayor_intv = tg.get_ayor()
    st = tg.xmin
    for wd in tg.get_sword():
        if wd.text.endswith('AYOR'):
            if abs(wd.xmin - st) > 0.001:
                ayor_intv.append(Interval('', st, wd.xmin))
            ayor_intv.append(Interval('AYOR', wd.xmin, wd.xmax))
            wd.text = wd.text[:-5]
            st = wd.xmax
    if tg.xmax - st > 0.001:
        ayor_intv.append(Interval('', st, tg.xmax))
    return tg

def del_sp123_(intvs):
    if intvs:
        for iv in intvs:
            if iv.text.startswith('sp'):
                iv.text = iv.text.replace('sp1', 'sp').replace('sp2', 'sp').replace('sp3', 'sp').replace('sp0', 'sp')

def del_sp123(tg):
    intvs =  tg.get_phoneme()
    del_sp123_(intvs)
    intvs = tg.get_mora()
    del_sp123_(intvs)
    intvs = tg.get_word()
    del_sp123_(intvs)
    intvs = tg.get_sword()
    del_sp123_(intvs)
    intvs = tg.get_trans()
    del_sp123_(intvs)
    return tg




def del_intnation(tg):
    for kana in tg.get_word():
        if kana.text[0] in ['＋', '，', '｜']:
            kana.text = kana.text[1:]
        if kana.text[-1] in ['．', '.', '？', '?', '！', '!']:
            kana.text = kana.text[:-1]
        if kana.text[-1] in ['．', '.', '？', '?', '！', '!']:
            kana.text = kana.text[:-1]

    return tg


def del_text(tg):
    txt = tg.get_trans(1)
    if len(txt.text)>3 and txt.text[3] == ':':
        txt.text = txt.text[:3]

    return tg

def cnvABC2NUM(ch):
    return ord(ch) - ord('a') +1

def cnvNUM2ABC(ch):
    return chr(ch+chr('a')-1)

def cnvFrame(tg):
    for fr in tg.get_frame():
        lbl = fr.text[1:]
        nn = cnvABC2NUM(fr.text[0])
        fr.text = f'{nn}:{lbl}'

    return tg

def cnvFrame_r(tg):
    for fr in tg.get_frame():
        ee = fr.text.split(':')
        nn = cnvNUM2ABC(ee[0])
        fr.text = f'{nn}{ee[1]}:{ee[2]}'

    return tg

def main(args):
    tg = TextGrid(args.file)

    tg = addMora(tg)
    tg = addAYOR(tg)

    tg = cnvFrame(tg)

    tg = del_sp123(tg)
    tg = del_intnation(tg)
    tg = del_text(tg)

    print(str(tg))

def main_rev(args):
    tg = TextGrid(args.file)

    tg.clear_mora()
    tg = delete_ayor(tg)

    tgo = cnvFrame_r(tg)

    print(str(tgo))

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    # parser.add_argument('-s', '--opt_str', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
    parser.add_argument('--reverse', '-r', action='store_true')
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

    if args.reverse:
        main_rev(args)
    else:
        main(args)
