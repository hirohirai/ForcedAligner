#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/05/09

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

import utils.Tts
import utils.TextGrid

# ログの設定
logger = logging.getLogger(__name__)


def mk_biphone_hist_cv_set(args):
    fn_dic = FnDic(args.dicSet)
    bi_all={}
    bi={}
    for kk in fn_dic.fset:
        bi[kk] = {}
    cons = set()
    vow = set()
    for ll in args.input:
        ee = ll.strip().split()
        ss = fn_dic[ee[2]]
        cons.add(ee[0])
        vow.add(ee[1])
        kk = (ee[0], ee[1])
        if kk in bi_all:
            bi_all[kk] += 1
            bi[ss][kk] += 1
        else:
            bi_all[kk] = 1
            for ky in bi.keys():
                if ky == ss:
                    bi[ss][kk] = 1
                else:
                    bi[ky][kk] = 0

    for cc in sorted(cons):
        for vv in sorted(vow):
            kk = (cc, vv)
            if kk in bi_all:
                for ky in bi.keys():
                    print(ky, cc, vv, bi_all[kk] - bi[ky][kk])
            '''
            else:
                for ky in bi.keys():
                    print(ky, cc, vv, 0)
            '''
def mk_biphone_hist_cv(args):
    bi = {}
    cons = set()
    vow = set()
    for ll in args.input:
        ee = ll.strip().split()
        cons.add(ee[0])
        vow.add(ee[1])
        kk = (ee[0], ee[1])
        if kk in bi:
            bi[kk] += 1
        else:
            bi[kk] = 1

    for cc in sorted(cons):
        for vv in sorted(vow):
            kk = (cc, vv)
            if kk in bi:
                print(cc, vv, bi[kk])
            else:
                print(cc, vv, 0)


def mk_biphone_hist(args):
    bi = {}
    cons = set()
    vow = set()
    for ll in args.input:
        ee = ll.strip().split()
        if ee[0].startswith('sp'):
            ee[0] = 'SP'
        if ee[0].endswith('|') and args.phr:
            ee[0] = ee[0][:-1]
        vow.add(ee[0])
        cons.add(ee[1])
        kk = (ee[0], ee[1])
        if kk in bi:
            bi[kk] += 1
        else:
            bi[kk] = 1

    for cc in sorted(cons):
        for vv in sorted(vow):
            kk = (vv, cc)
            if kk in bi:
                print(vv, cc, bi[kk])
            else:
                print(vv, cc, 0)


class FnDic:
    def __init__(self, fn):
        self.body={}
        self.fset = set()
        with open(fn) as ifs:
            for ll in ifs:
                ee = ll.strip().split()
                self.body[ee[0]] = ee[1]
                self.fset.add(ee[1])

    def __getitem__(self, ix):
        return self.body[ix]



def mk_biphone_hist_set(args):
    fn_dic = FnDic(args.dicSet)
    bi_all={}
    bi={}
    for kk in fn_dic.fset:
        bi[kk] = {}
    cons = set()
    vow = set()
    for ll in args.input:
        ee = ll.strip().split()
        ss = fn_dic[ee[2]]
        if ee[0].startswith('sp'):
            ee[0] = 'SP'
        if ee[0].endswith('|') and args.phr:
            ee[0] = ee[0][:-1]
        vow.add(ee[0])
        cons.add(ee[1])
        kk = (ee[0], ee[1])
        if kk in bi_all:
            bi_all[kk] += 1
            bi[ss][kk] += 1
        else:
            bi_all[kk] = 1
            for ky in bi.keys():
                if ky == ss:
                    bi[ss][kk] = 1
                else:
                    bi[ky][kk] = 0


    for cc in sorted(cons):
        for vv in sorted(vow):
            kk = (vv, cc)
            if kk in bi_all:
                for ky in bi.keys():
                    print(ky, vv, cc, bi_all[kk] - bi[ky][kk])


def mk_biphone_list(args):
    for ll in args.input:
        fn_ = ll.strip().split()
        if args.verbose:
            print(fn_)
        mk_biphone(fn_[0])


def mk_biphone_list_cv(args):
    for ll in args.input:
        fn_ = ll.strip().split()
        if args.verbose:
            print(fn_)
        mk_biphone_cv(fn_[0])


def print_cv(cc, vv, fn=''):
    if vv[0] in 'AIUEO':
        vv = vv.lower()
    if cc[-1] == '_':
        cc = cc[:-1]
    print(f'{cc} {vv} {fn}')


def print_bi(vv, cc, fn=''):
    if vv[0] in 'AIUEO':
        vv = vv.lower()
    if cc[0] in 'AIUEO':
        cc = cc.lower()
    if len(vv)>1 and vv[-1] == 'j':
        vv = vv[:-1]
    if len(cc)>1 and cc[-1] == 'j':
        cc = cc[:-1]
    if cc[-1] == '_':
        cc = cc[:-1]
    print(f'{vv} {cc} {fn}')

def mk_biphone(fn):
    fn_ = f'{args.dir}/{fn}.TextGrid'
    tg = utils.TextGrid.TextGrid(fn_)
    tts = utils.Tts.textGrid_to_Tts(tg)
    for ix_p, phr in enumerate(tts.phrases):
        for ix_w, wd in enumerate(phr.words):
            for ix_m, mr in enumerate(wd.moras):
                if ix_m ==0:
                    if ix_p == 0 and ix_w == 0:
                        if mr.cons == '':
                            print_bi('sp', mr.vow, fn)
                        else:
                            print_bi('sp', mr.cons, fn)
                    elif ix_w > 0 and mr.vow[0] in 'AIUEOaiueo':
                        lvv = phr.words[ix_w - 1].moras[-1].vow if phr.words[ix_w - 1].moras[-1].vow != 'H' else \
                        phr.words[ix_w - 1].moras[-2].vow
                        if mr.cons != '':
                            print_bi(lvv + '|', mr.cons, fn)
                        else:
                            print_bi(lvv + '|', mr.vow, fn)
                    elif ix_p > 0 and mr.vow[0] in 'AIUEOaiueo':
                        lwrd = tts.phrases[ix_p-1].words[-1]
                        lvv = lwrd.moras[-1].vow if lwrd.moras[-1].vow != 'H' else lwrd.moras[-2].vow
                        if mr.cons != '':
                            print_bi(lvv + '|', mr.cons, fn)
                        else:
                            print_bi(lvv + '|', mr.vow, fn)

                else:
                    if mr.cons == '':
                        if mr.vow == 'H':
                            print_bi(wd.moras[ix_m-1].vow, wd.moras[ix_m-1].vow, fn)
                        elif wd.moras[ix_m-1].vow == 'H':
                            print_bi(wd.moras[ix_m-2].vow, mr.vow, fn)
                        else:
                            print_bi(wd.moras[ix_m-1].vow, mr.vow, fn)
                    else:
                        if wd.moras[ix_m-1].vow == '':
                            print_bi(wd.moras[ix_m - 1].cons, mr.cons, fn)
                        elif wd.moras[ix_m-1].vow == 'H':
                            print_bi(wd.moras[ix_m - 2].vow, mr.cons, fn)
                        else:
                            print_bi(wd.moras[ix_m - 1].vow, mr.cons, fn)

def mk_biphone_cv(fn):
    fn_ = f'{args.dir}/{fn}.TextGrid'
    tg = utils.TextGrid.TextGrid(fn_)
    tts = utils.Tts.textGrid_to_Tts(tg)
    for ix_p, phr in enumerate(tts.phrases):
        for ix_w, wd in enumerate(phr.words):
            for ix_m, mr in enumerate(wd.moras):
                if len(mr.cons) > 0 and len(mr.vow) > 0:
                    print_cv(mr.cons, mr.vow, fn)


def fname(args):
    for ll in args.input:
        ee = ll.strip().split()
        for eee in ee[1:]:
            print(f'{ee[0].upper()}{int(eee):02}')


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
    parser.add_argument('--dicSet', default='')
    parser.add_argument('--mode', '-m', default='')
    parser.add_argument('--phr', action='store_true')
    parser.add_argument('--dir', default='atr503/TextGrid')
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

    if args.mode == 'fname':
        fname(args)
    elif args.mode == 'bilist':
        mk_biphone_list(args)
    elif args.mode == 'cvlist':
        mk_biphone_list_cv(args)
    elif args.mode == 'cvhist':
        if len(args.dicSet) > 0:
            mk_biphone_hist_cv_set(args)
        else:
            mk_biphone_hist_cv(args)
    else:
        if len(args.dicSet)>0:
            mk_biphone_hist_set(args)
        else:
            mk_biphone_hist(args)
