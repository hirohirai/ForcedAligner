#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/03/17

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

import numpy as np
import scipy.special

import utils.TextGrid
import utils.text.kana2rtMRI
from utils.Tts import Tts, tts_to_textGrid, set_default_join
from utils.text.rtMRI import rom_to_id
from utils.dur_stats import DurLogLikelihood

# ログの設定
logger = logging.getLogger(__name__)


class Label:
    def __init__(self, phn_):
        self.phn = phn_
        self.num = rom_to_id[phn_]

    def __str__(self):
        return f'{self.phn}:{self.num}'

class Unit:
    def __init__(self, Ix):
        self.con = None
        self.edge = None
        self.cost = 0.0
        self.best_cost = 0.0 # ここで終わる時の最適なパスのコスト
        self.best_len = 0
        self.labelIx = Ix

    def __str__(self):
        return f'{self.labelIx}: {self.cost}  {self.best_cost} {self.best_len}'


def get_tts(tg):
    if isinstance(tg, str):
        tg = utils.TextGrid.TextGrid(tg)
    ks = utils.text.kana.KanaSent()
    kana = tg.get_jeitaKana()
    ks.set_kana(kana)
    if tg.get_sword() is not None:
        swrds = []
        for sw in tg.get_sword():
            if len(sw.text) > 0 and not sw.text.startswith('sp') and sw.text != '#':
                swrds.append(sw.text)
        ks.add_sword(swrds)
    tts = Tts()
    tts.from_kanaSent(ks)
    tts.xmin = tg.xmin
    tts.xmax = tg.xmax
    tts.text = tg.get_text()
    sted_time = tg.getStEd('trans')
    if sted_time[0]<0:
        sted_time = tg.getStEd()

    return tts, sted_time

def get_labels(tg):
    if isinstance(tg, str):
        tg = utils.TextGrid.TextGrid(tg)
    tts, sted_time = get_tts(tg)
    labels = [Label('sp'), ]
    last_vow = ''
    for phr in tts.phrases:
        for wd in phr.words:
            for mr in wd.moras:
                if mr.cons:
                    if mr.hasCL:
                        labels.append(Label('<cl>'))
                    labels.append(Label(mr.cons))
                    if mr.hasJ:
                        labels.append(Label('<j>'))
                if mr.vow:
                    if mr.vow == 'H':
                        labels.append(Label(last_vow+'H'))
                    else:
                        labels.append(Label(mr.vow))
                        last_vow = mr.vow
    labels.append(Label('sp'))
    return labels, tts, sted_time

def create_network(labels, cost, dur_cost, width=0.005, p_len=0.3, s_len=1.5, dur_w=1.0):
    s_width = round(s_len / width)
    p_width = round(p_len / width)
    sz = cost.shape[0]
    unt = Unit(0)
    unt.cost = cost[0,0]
    unt.best_cost = unt.cost + dur_cost.get_LogLikelihood('sp', width)
    ulist = [None,]*len(labels)
    ulist[0] = unt
    for ii in range(1, sz):
        nn = ii+1 if ii+1 < len(labels) else len(labels)
        for jj in range(nn-1, -1, -1):
            unt = Unit(jj)
            unt.con = ulist[jj]
            unt.cost = cost[ii, labels[jj].num]
            if jj>0:
                unt.edge = ulist[jj-1]
                blen = 0
                bcost = unt.edge.best_cost + unt.cost + dur_cost.get_LogLikelihood(labels[unt.labelIx].phn, width) * dur_w
                tmpunt = unt.con
                addcost = unt.cost
                ed = ii+1
                if labels[unt.labelIx].phn.startswith('sp') or labels[unt.labelIx].phn == '#':
                    if ed > s_width:
                        ed = s_width
                else:
                    if ed > p_width:
                        ed = p_width
                for kk in range(1, ed):
                    if tmpunt is None:
                        break
                    addcost += tmpunt.cost
                    tmpcost = tmpunt.edge.best_cost + addcost + dur_cost.get_LogLikelihood(labels[unt.labelIx].phn, (kk+1)*width) * dur_w
                    if bcost < tmpcost:
                        bcost = tmpcost
                        blen = kk
                    tmpunt = tmpunt.con
                unt.best_cost = bcost
                unt.best_len = blen
            else:
                unt.best_len = ii
                bcost = unt.cost
                tmpunt= unt.con
                for kk in range(ii):
                    bcost += tmpunt.cost
                    tmpunt = tmpunt.con
                unt.best_cost = bcost + dur_cost.get_LogLikelihood(labels[unt.labelIx].phn, (unt.best_len+1)*width) * dur_w

            ulist[jj] = unt

    return ulist[-1]


def set_time(labels, tts, nets, st):
    times = [0.0] * len(labels)
    for unt in nets:
        times[unt.labelIx] += 0.005

    st_times = [st]
    for ti in times:
        st_times.append(st_times[-1] + ti)

    ix= 1
    for phr in tts.phrases:
        for wd in phr.words:
            for mr in wd.moras:
                mr.st = st_times[ix]
                if mr.cons:
                    if mr.hasCL:
                        if '<cl>' !=  labels[ix].phn:
                            logger.error(f'<cl> {labels[ix].phn} {times[ix]}')
                        mr.CL_len = times[ix]
                        ix += 1
                        if mr.cons != labels[ix].phn:
                            logger.error(f'{mr.cons} {labels[ix].phn} {times[ix]}')
                    mr.clen = times[ix]
                    ix += 1
                    if mr.hasJ:
                        if '<j>' != labels[ix].phn:
                            logger.error(f'<j> {labels[ix].phn} {times[ix]}')
                        mr.J_len = times[ix]
                        ix += 1
                if mr.vow:
                    if mr.vow == 'H':
                        if 'H' != labels[ix].phn[-1]:
                            logger.error(f'H {labels[ix].phn} {times[ix]}')
                        mr.vlen = times[ix]
                        ix += 1
                    else:
                        if mr.vow != labels[ix].phn:
                            logger.error(f'{mr.vow} {labels[ix].phn} {times[ix]}')
                        mr.vlen = times[ix]
                        ix += 1


    set_default_join(tts)
    tg = tts_to_textGrid(tts)

    return tg


def back_track(last_unit):
    unt = Unit(-1)
    unt.edge = last_unit
    lunit = []
    while unt.edge:
        unt = unt.edge
        lunit.append(unt)
        for ii in range(unt.best_len):
            unt = unt.con
            lunit.append(unt)

    lunit.reverse()

    return lunit


def main(args):
    tgo = utils.TextGrid.TextGrid(args.tgFile)
    labels, tts, sted_time = get_labels(tgo)

    cost = np.load(args.costFile)
    cost = np.log(scipy.special.softmax(cost, axis=1))

    if args.ed_lbl>0:
        labels = labels[args.st_lbl:args.ed_lbl]
    else:
        labels = labels[args.st_lbl:]
    if args.debug:
        for ix, lb in enumerate(labels):
                print(ix, lb)

    st_time = sted_time[0] - args.st_width if args.st_time<-0.000000001 else args.st_time
    ed_time = sted_time[1] + args.st_width if args.ed_time<-0.000000001 else args.ed_time

    if st_time<0.0:
        st_time = 0.0
        st_ix = 0
    else:
        st_ix = int(round(st_time / 0.005))
        st_time = st_ix*0.005

    ed_ix = int(round(ed_time / 0.005)) + 1
    if ed_ix == 1 or ed_ix > len(cost):
        ed_ix = len(cost)
    ed_time = ed_ix * 0.005

    cost = cost[st_ix:ed_ix]


    dur_cost = DurLogLikelihood(fn=args.durdic)
    last_unit = create_network(labels, cost, dur_cost, p_len=args.p_len, s_len=args.s_len, dur_w=args.dur_weight)
    nets = back_track(last_unit)

    if args.debug:
        for ix, nn in enumerate(nets):
            print(ix, labels[nn.labelIx].phn,nn.labelIx)
    else:
        tg = set_time(labels, tts, nets, st_time)
        tg.copyFrameNum(tgo)
        print(tg)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('tgFile')
    parser.add_argument('costFile')
    parser.add_argument('--dur_weight', type=float, default=10.0)
    parser.add_argument('--st_width', type=float, default=0.0)
    parser.add_argument('--p_len', type=float, default=0.3)
    parser.add_argument('--s_len', type=float, default=1.5)
    parser.add_argument('--durdic', default='data/stats/dur_dic.pkl')
    parser.add_argument('--st_lbl', type=int, default=0)
    parser.add_argument('--ed_lbl', type=int, default=-1)
    parser.add_argument('--st_time', type=float, default=-1.0)
    parser.add_argument('--ed_time', type=float, default=-1.0)
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
