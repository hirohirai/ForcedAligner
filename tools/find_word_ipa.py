#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/05/23

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import sexpdata

import utils.text.kana2rtMRI

# ログの設定
logger = logging.getLogger(__name__)


class Word:
    def __init__(self,mi, prn, rom):
        self.midashi = mi
        self.pron = prn
        self.rom = rom

    def __str__(self):
        return f'{self.midashi} {self.pron}'

def get_word(ifs):
    words = []
    for ll in ifs:
        dat = f'({ll.strip()})'
        try:
            sdat = sexpdata.loads(dat)
            for ee in sdat[1]:
                if ee[0][:] == '見出し語':
                    midashi = ee[1][0][:]
                if ee[0][:] == '発音':
                    pron = ee[1][:]
            roms = utils.text.kana2csj.kana2roms(pron)
            words.append(Word(midashi, pron, roms))
        except:
            pass

    return words

def conv_vv(vv):
    if vv in 'AIUEO':
        vv = vv.lower()

    return vv

def conv_cc(cc):
    if len(cc)>1 and cc[-1] == 'j':
        cc = cc[:-1]
    return cc


def search_vc(args):
    (s_v, s_c) = args.vc.split(':')
    words = get_word(args.input)
    for wd in words:
        for ix in range(1, len(wd.rom)):
            vv = conv_vv(wd.rom[ix-1][1])
            cc = conv_cc(wd.rom[ix][0])
            if s_v == vv and s_c == cc:
                print(wd)

def search_cv(args):
    (s_c, s_v) = args.cv.split(':')
    words = get_word(args.input)
    for wd in words:
        for ix in range(0, len(wd.rom)):
            cc = wd.rom[ix][0]
            vv = conv_vv(wd.rom[ix][1])
            if s_v == vv and s_c == cc:
                print(wd)

def search_cc(args):
    (s_c1, s_c2) = args.cc.split(':')
    words = get_word(args.input)
    for wd in words:
        for ix in range(1, len(wd.rom)):
            if len(wd.rom[ix-1][1])>0:
                continue
            cc1 = conv_cc(wd.rom[ix-1][0])
            cc2 = conv_cc(wd.rom[ix][0])
            if s_c1 == cc1 and s_c2 == cc2:
                print(wd)

def search_vv(args):
    (s_v1, s_v2) = args.vv.split(':')
    words = get_word(args.input)
    for wd in words:
        for ix in range(1, len(wd.rom)):
            if len(wd.rom[ix][0])>0:
                continue
            vv1 = conv_vv(wd.rom[ix-1][1])
            vv2 = conv_vv(wd.rom[ix][1])
            if s_v1 == vv1 and s_v2 == vv2:
                print(wd)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
    parser.add_argument('--vc', default='')
    parser.add_argument('--cc', default='')
    parser.add_argument('--vv', default='')
    parser.add_argument('--cv', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
    parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
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

    if args.vv:
        search_vv(args)
    if args.vc:
        search_vc(args)
    if args.cc:
        search_cc(args)
    if args.cv:
        search_cv(args)
