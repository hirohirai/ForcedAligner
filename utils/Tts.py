#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Tts.py
    Author: hirai
    Data: 2022/03/24
"""

import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging

import numpy as np

import utils.TextGrid as TextGrid
from utils.text.kana2rtMRI import kana2roms
from utils.text.rtMRI2kana import rom2kana_c
from utils.text.rtMRI import CL_ph, J_ph
#from utils.text.kana2rom import kana2roms
#from utils.text.rom2kana import rom2kana_c
import utils.text.kana

logger = logging.getLogger(__name__)

class TtsMora:
    def __init__(self, ph=['', ''], dur_=[0, 0, 0, 0], f0_=[0, 0, 0, 0, 0], join_=[False,False,False,False], st_=0):
        self.cons = ph[0]
        self.vow = ph[1]  # sp: pause

        if self.cons in CL_ph:
            self.hasCL = True
            self.CL_len = dur_[0]
        else:
            self.hasCL = False
            self.CL_len = 0.0
        self.clen = dur_[1]  # 子音、母音　sec
        self.joinCL = join_[0]
        self.join_C = join_[1] #子音が前の音と結合している。
        if self.cons in J_ph:
            self.hasJ = True
            self.J_len = dur_[3]
        else:
            self.hasJ = False
            self.J_len = 0.0
        self.vlen = dur_[2]  # 子音、母音　sec
        self.join_V = join_[2] #母音が前の音と結合している。
        self.joinJ = join_[3]
        self.F0 = list(f0_)  # Hz
        self.st = st_  # 学習用　ファイルの先頭からの位置
        self.vF0 = []

    def get_st(self):
        return self.st

    def get_clen(self):
        return self.CL_len + self.clen + self.J_len

    def get_vlen(self):
        return self.vlen

    def get_ced(self):
        return self.st + self.CL_len + self.clen + self.J_len

    def get_vst(self):
        return self.st + self.CL_len + self.clen + self.J_len

    def get_ed(self):
        return self.st + self.clen + self.vlen + self.CL_len + self.J_len

    def get_kana(self):
        if self.vow.startswith('sp'):
            return self.vow
        else:
            return rom2kana_c(self.cons, self.vow)

    def div_all(self):
        self.joinJ = self.joinCL = self.join_V = self.join_C = False

    def __str__(self):
        ff = ''
        for f_ in self.F0:
            ff += f' {f_:.3g},'
        outs = f'({self.cons}, {self.vow}), [{self.CL_len:.3f}, {self.clen:.3f}, {self.vlen:.3f}, {self.J_len:.3f}], [{self.joinCL}, {self.join_C}, {self.join_V}, {self.joinJ}], [{ff[1:-1]}], {self.st:.3f}, {self.hasCL},  {self.hasJ}\n'
        return outs


B_DIV_SYM=['', '，', '｜']

class TtsWord:
    def __init__(self):
        self.moras = []
        self.sword = []
        self.bound_div = 0  # 前との関係　0:通常の下降　2:フレーズの境界か強調による上昇　1:並列
        self.acc_up = 1  # 上昇位置
        self.acc_down = 0  # 下降位置
        self.acc_level = 1  # アクセントのレベル　0:弱　1:通常　2:強
        self.hasPause = 0  # ポーズの終わる位置

    def get_sword(self, ix):
        if ix > len(self.sword):
            return None
        st = 0 if ix-1 < 0 else self.sword[ix-1]
        st += self.hasPause
        ed = self.sword[ix]+self.hasPause if ix < len(self.sword) else len(self.moras)
        kana = ''
        for ii in range(st, ed):
            kana += self.moras[ii].get_kana()
        return kana, self.moras[st].get_st(), self.moras[ed-1].get_ed()

    def set_hasPause(self):
        for ix, mr in enumerate(self.moras):
            if not mr.vow.startswith('sp'):
                self.hasPause = ix
                break

    def get_st(self):
        return self.moras[self.hasPause].get_st()

    def get_ed(self):
        return self.moras[-1].get_ed()

    def get_kana(self, stp=0):
        kana = B_DIV_SYM[self.bound_div]
        stp = 1 if self.moras[0].vow.startswith('sp') else 0
        for ix, mr in enumerate(self.moras[stp:]):
            ka = mr.get_kana()
            if ka == 'ー':
                if mr.vow[0] != self.moras[stp+ix].vow[0]:
                    mr.vow = mr.vow[:-1]
                    ka = mr.get_kana()
            kana += ka
            if ix+1 == self.acc_down:
                if self.acc_level != 1:
                    kana += f'’{self.acc_level}'
                else:
                    kana += '’'

        return kana

    def __str__(self):
        outs = f'#[WORD] b_div: {self.bound_div}  a_up: {self.acc_up}  a_down: {self.acc_down}  a_level: {self.acc_level}  sword:'
        for sw in self.sword:
            outs += f' {sw}'
        outs += '\n'
        for mr in self.moras:
            outs += str(mr)
        return outs

class TtsPhrase:
    def __init__(self):
        self.words = []
        self.bound_end = ''  # 直後の終端記号 半角 '':なし .:通常の終端 ?:上昇疑問 !:断定など急激な終端 !?:付加疑問など下がって上がる

    def __str__(self):
        outs = f'#[PHRASE] b_end: {self.bound_end}\n'
        for wrd in self.words:
            outs += str(wrd)
        return outs

class TgPE:
    def __init__(self, xmin, xmax, txt, jn=False):
        self.xmin = xmin
        self.xmax = xmax
        self.text = txt
        self.join = jn # 前と結合するか

    def __str__(self):
        outs = f'{self.text}, {self.xmin}, {self.xmax}, {self.join}'
        return outs

class TgPhoneme(list):
    def __init__(self, tgphns):
        super().__init__()
        for tp in tgphns:
            self.append(TgPE(tp.xmin, tp.xmax, tp.text))

    def div_ph(self, text, ltxt, ntxt):
        eee = text.strip().split(',')
        if '<cl>' in eee:
            eee_ = []
            for ix, ee in enumerate(eee):
                if ix > 0 and ee == '<cl>' and eee_[-1] in ['#', 'sp', 'sp0', 'sp1', 'sp2', 'sp3']:
                    eee_[-1] = f'{eee_[-1]},<cl>'
                else:
                    eee_.append(ee)
            eee = eee_

        durs = []
        neee = ntxt.strip().split(',')
        ntxt = neee[0]
        for ix, ee in enumerate(eee):
            nee = eee[ix+1] if ix < len(eee)-1 else ntxt
            if len(ltxt)>0 and ltxt[-1] in 'aiueoNAIUEO' and ee in 'aiueoNAIUEO':
                durs.append(2.0)
            elif ee == '<cl>' and ltxt != 'Q':
                durs.append(0.2)
            #elif ltxt == '<cl>' and len(durs)>0 and durs[-1] <= 0.5001:
            elif ltxt == '<cl>':
                durs.append(0.6)
            else:
                durs.append(1.0)
            ltxt = ee

        return eee, durs

    def expand(self):
        for ix in range(len(self)-1,-1,-1):
            ee = self[ix]
            ltxt = '' if ix==0 else self[ix-1].text
            ntxt = '' if ix==len(self)-1 else self[ix+1].text
            if ',' in ee.text:
                phs,rdurs = self.div_ph(ee.text, ltxt, ntxt)
                if len(phs) > 1:
                    alrdurs = 0.0
                    for rd in rdurs:
                        alrdurs += rd
                    udur = (ee.xmax - ee.xmin) / alrdurs
                    st = ee.xmin
                    ed = st + udur*rdurs[0]
                    self[ix].xmin = st
                    self[ix].xmax = ed
                    self[ix].text = phs[0]
                    st = ed
                    for ii, ph in enumerate(phs[1:], 1):
                        ed = st + udur * rdurs[ii]
                        self.insert(ix+ii, TgPE(st, ed, ph, True))
                        st = ed

        for ix in range(len(self)-1,-1,-1):
            if ',' in self[ix].text:
                ee = self[ix]
                phs = ee.text.split(',')
                if phs[1] == '<cl>':
                    ee.text = phs[0]
                    ed = ee.xmax
                    dur = ee.xmax - ee.xmin - 0.02
                    if dur < 0.0:
                        dur = (ee.xmax - ee.xmin) * 0.9
                    ee.xmax = ee.xmin + dur
                    self.insert(ix+1,TgPE(ee.xmax, ed, phs[1], True))
                else:
                    logger.error(f'text error: {ee.text}')
                    raise Exception('text error')



class Tts:
    def __init__(self, ifs=None):
        self.phrases = []
        self.words = []
        self.moras = []
        self.text = ''
        self.xmin = 0.0
        self.xmax = 0.0
        self.fr_st = 0.0
        self.fr_fps = 0.0
        self.fr_fn = ''
        if ifs:
            for ll in ifs:
                ll = ll.strip()
                if len(ll) == 0:
                    continue
                if ll[0] == '#':
                    if ll.startswith('#[SENT]'):
                        ee = ll.strip().split()
                        if len(ee) > 3:
                            self.xmin = float(ee[1])
                            self.xmax = float(ee[2])
                            self.text = ee[3]
                            if len(ee) > 6:
                                if ee[-3] == 'FRAME:':
                                    self.text = ' '.join(ee[3:-3])
                                    self.fr_st = float(ee[-2])
                                    self.fr_fps = float(ee[-1])
                                    self.fr_fn = ''
                                if ee[-4] == 'FRAME:':
                                    self.text = ' '.join(ee[3:-4])
                                    self.fr_st = float(ee[-3])
                                    self.fr_fps = float(ee[-2])
                                    self.fr_fn = float(ee[-1])


                    elif ll.startswith('#[PHRASE]'):
                        b_end = ll.split(':')[1].strip()
                        self.phrases.append(TtsPhrase())
                        self.phrases[-1].bound_end = b_end
                    elif ll.startswith('#[WORD]'):
                        ee = ll[7:].strip().split()
                        b_div = int(ee[1])
                        a_up = int(ee[3])
                        a_down = int(ee[5])
                        a_level = int(ee[7])
                        self.words.append(TtsWord())
                        self.words[-1].bound_div = b_div
                        self.words[-1].acc_up = a_up
                        self.words[-1].acc_down = a_down
                        self.words[-1].acc_level = a_level
                        for sw in ee[9:]:
                            self.words[-1].sword.append(int(sw))
                        self.phrases[-1].words.append(self.words[-1])
                else:
                    ee = ll.replace('(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
                    cons = ee[0].strip()
                    vows = ee[1].strip()
                    cl_len = float(ee[2].strip())
                    clen = float(ee[3].strip())
                    vlen = float(ee[4].strip())
                    j_len = float(ee[5].strip())
                    joinCL = True if ee[6] == 'True' else False
                    join_C = True if ee[7] == 'True' else False
                    join_V = True if ee[8] == 'True' else False
                    joinJ = True if ee[9] == 'True' else False
                    f0 = [float(x) for x in ee[10:15]]
                    st = float(ee[15])
                    hasCL = True if ee[16] == 'True' else False
                    hasJ = True if ee[17] == 'True' else False
                    self.words[-1].moras.append(TtsMora([cons,vows], [clen,vlen, cl_len, j_len], f0, [joinCL, join_C, join_V, joinJ], st))

            self.set_hasPause()

    def set_hasPause(self):
        for phr in self.phrases:
            for wd in phr.words:
                wd.set_hasPause()

    def add_pause(self, kwrd):
        if kwrd.bound_pau == '＿0':
            if not self.moras[-1].vow.startswith('sp'):
                self.moras.append(TtsMora(['', 'sp0']))
                self.words[-1].moras.append(self.moras[-1])
        else:
            if kwrd.bound_pau == '＿2':
                if not self.moras[-1].vow.startswith('sp'):
                    self.moras.append(TtsMora(['', 'sp2']))
                    self.words[-1].moras.append(self.moras[-1])
                else:
                    self.moras[-1].vow = 'sp2'
            else:
                if not self.moras[-1].vow.startswith('sp'):
                    self.moras.append(TtsMora(['', 'sp1']))
                    self.words[-1].moras.append(self.moras[-1])
                elif kwrd.bound_pau == '＿0':
                    self.moras[-1].vow = 'sp1'
        self.words[-1].hasPause = len(self.words[-1].moras)

    def add_mora(self, kwrd, cwrd):
        cwrd.acc_up = kwrd.accup
        cwrd.acc_down = kwrd.accdown
        cwrd.acc_level = kwrd.acclevel
        if kwrd.bound_div == '｜':
            cwrd.bound_div = 2
        elif kwrd.bound_div == '，':
            cwrd.bound_div = 1

        phs = kana2roms(''.join(kwrd.phs))
        for ph in phs:
            self.moras.append(TtsMora(ph))
            cwrd.moras.append(self.moras[-1])

    def from_kanaSent(self, kana_sent):
        self.phrases.append(TtsPhrase())
        for wd_ix, kwrd in enumerate(kana_sent.words):
            if len(self.words) == 0 or len(self.words[-1].moras) > 1 or not self.words[-1].moras[-1].vow.startswith('sp'):
                self.words.append(TtsWord())
            if len(kwrd.phs) == 0:
                if len(self.moras) > 0:  # 文の先頭以外で、無音だけのWordがあった時後ろに纏める
                    self.add_pause(kwrd)
                if kwrd.bound_div == '｜':
                    self.words[-1].bound_div = 2
                elif kwrd.bound_div == '，' and self.words[-1].bound_div < 1:
                    self.words[-1].bound_div = 1
                continue
            for sw in kwrd.sword:
                self.words[-1].sword.append(sw)
            if kwrd.bound_pau:
                self.add_pause(kwrd)

            if len(self.moras)>0:
                if self.moras[-1].vow == 'sp1' or self.moras[-1].vow == 'sp2' or self.phrases[-1].bound_end or kwrd.bound_div == '｜':
                    if len(self.phrases[-1].words) > 0:
                        self.phrases.append(TtsPhrase())

            self.add_mora(kwrd, self.words[-1])


            self.phrases[-1].words.append(self.words[-1])

            if kwrd.bound_end:
                self.phrases[-1].bound_end = kwrd.bound_end.replace('．', '.').replace('？', '?').replace('！', '!')
        self.set_hasPause()

    def set_f0(self, f0, f0r=None):
        nn = 5
        for phr in self.phrases:
            for wrd in phr.words:
                for mr in wrd.moras:
                    vst = round(mr.get_vst() / 0.005)
                    ved = round(mr.get_ed() / 0.005)
                    ll = (ved - vst) / nn
                    dd = 0.0 if ll >= 1 else 1.0
                    for ii in range(nn):
                        st = vst + ll * ii
                        ed = st + ll + dd
                        ix = np.arange(st, ed, dtype=int) + 5 # worldとのずれ後で検証
                        ix = ix[ix<len(f0)]
                        mrf0 = f0[ix]
                        if len(mrf0) == 0:
                            logger.debug(f'{mr.get_vst()}:{mr.get_ed()} No F0 ix')
                        maxf0 = max(mrf0)
                        f0_ = np.mean(maxf0[maxf0>0]) if maxf0 > 0 else 0.0
                        if f0r is not None:
                            zf0 = f0r[ix]
                            zf0[zf0>0] = 1
                            f0_ = -f0_ if np.mean(zf0) < 0.3 else f0_
                        mr.F0[ii] = f0_

    def comp_cons(self, ph1, ph2):
        if ph1 == ph2:
            return True
        if ph1 == 'f' and ph2 == 'h':
            return True

        return False


    def set_time(self, tgrid, force_clJ=False):
        self.xmin = tgrid.xmin
        self.xmax = tgrid.xmax
        tphn = TgPhoneme(tgrid.get_phoneme())
        tphn.expand()
        for tix,  ph in enumerate(tphn):
            if ph.text and ph.text != 'sp' and ph.text != '#':
               break

        for phr in self.phrases:
            for wrd in phr.words:
                for mr in wrd.moras:
                    logging.debug(f'mr')
                    if mr.vow.startswith('sp'):
                        if tix>0 and (tphn[tix-1].text.startswith('sp') or tphn[tix-1].text == '#'):
                            mr.st = tphn[tix-1].xmin
                            mr.vlen = tphn[tix-1].xmax - tphn[tix-1].xmin
                        elif tphn[tix].text.startswith('sp') or tphn[tix].text == '#':
                            mr.st = tphn[tix].xmin
                            mr.vlen = tphn[tix].xmax - tphn[tix].xmin
                            tix += 1
                        else:
                            logging.error(f'NO SP: {tphn[tix-1].text}, {tphn[tix-1].text}, {tphn[tix].text}, {tphn[tix+1].text} ')
                            raise Exception('NO SP')
                        continue
                    if mr.cons:
                        if tphn[tix].text == '<cl>':
                            mr.CL_len = tphn[tix].xmax - tphn[tix].xmin
                            mr.joinCL = tphn[tix].join
                            tix += 1
                        elif mr.hasCL and force_clJ:
                            mr.CL_len = (tphn[tix].xmax - tphn[tix].xmin) /3
                            tphn[tix].xmin += mr.CL_len
                        if force_clJ or self.comp_cons(mr.cons,tphn[tix].text):
                            mr.st = tphn[tix].xmin - mr.CL_len
                            mr.clen = tphn[tix].xmax - tphn[tix].xmin
                            mr.join_C = tphn[tix].join
                            tix += 1
                        else:
                            logging.error(f'NOT EQ {mr} and {tphn[tix]}')
                            raise Exception('NOT EQ')
                        if tphn[tix].text == '<j>':
                            mr.J_len = tphn[tix].xmax - tphn[tix].xmin
                            mr.joinJ = tphn[tix].join
                            tix += 1
                        elif mr.hasJ and force_clJ:
                            mr.J_len = mr.clen/3
                            mr.clen -= mr.J_len

                    if mr.vow:
                        if mr.vow == tphn[tix].text or mr.vow == tphn[tix].text.lower():
                            mr.vow = tphn[tix].text
                            if mr.st == 0:
                                mr.st = tphn[tix].xmin
                            mr.vlen = tphn[tix].xmax - tphn[tix].xmin
                            mr.join_V = tphn[tix].join
                            tix += 1
                        elif mr.vow == tphn[tix].text[0] or mr.vow == tphn[tix].text[0].lower() or mr.vow == 'H': #前との境界が不鮮明のとき
                            if mr.vow != 'H':
                                mr.vow = tphn[tix].text
                            if mr.st == 0:
                                mr.st = tphn[tix].xmin
                            mr.vlen = tphn[tix].xmax - tphn[tix].xmin
                            mr.join_V = True
                            tix += 1
                        else:
                            logging.error(f'NOT EQ {mr} and {tphn[tix]}')
                            raise Exception('NOT EQ')

                    while tix < len(tphn):
                        if tphn[tix].text and tphn[tix].text != 'sp' and tphn[tix].text != '#':
                            break
                        tix += 1

    def get_time(self):
        outs = []
        for phr in self.phrases:
            for wrd in phr.words:
                for mr in wrd.moras:
                    if mr.cons:
                        outs.append((mr.st, mr.CL_len + mr.clen + mr.J_len))
                    if mr.vow:
                        outs.append((mr.st + mr.CL_len + mr.clen + mr.J_len, mr.vlen))
        return outs

    def __str__(self):
        outs = f'#[SENT] {self.xmin} {self.xmax} {self.text} FRAME: {self.fr_st} {self.fr_fps} {self.fr_fn}\n'
        for phr in self.phrases:
            outs += str(phr)
        return outs

    def lv_HtoXH(self):
        lastV = ''
        for phr in self.phrases:
            for wrd in phr.words:
                for mr in wrd.moras:
                    if mr.vow:
                        if mr.vow == 'H':
                            if lastV[0] in 'aiueoAIUEO':
                                mr.vow = lastV[0] + 'H'
                        else:
                            lastV = mr.vow
                    else:
                        lastV = ''
        return self

    def div_all(self):
        for phr in self.phrases:
            for wrd in phr.words:
                for mr in wrd.moras:
                    mr.div_all()

def textGrid_to_Tts(tg, f0=None, f0r=None):
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
    tts.set_time(tg)
    if f0 is not None:
        tts.set_f0(f0, f0r)

    tts.xmin = tg.xmin
    tts.xmax = tg.xmax
    tts.text = tg.get_text()

    fr = tg.get_frame()
    if fr:
        fr_width = 1000000 / round(1000000 / (fr[1].xmax - fr[1].xmin))
        tts.fr_fps = 1 / fr_width
        if abs(tts.fr_fps - 27.1739) < 0.0001:
            tts.fr_fps = 27.1739
            fr_width = 1/tts.fr_fps
        fre = fr[1].text.split(':')
        frnum = int(fre[-1].strip())
        tts.fr_st = (frnum-1) * fr_width - fr[1].xmin
        if fr[0].xmin == 0.0:
            tmpst = round(tts.fr_st / fr_width) * fr_width
            if abs(tmpst - tts.fr_st) < 0.000001:
                tts.fr_st = tmpst
        tts.fr_fn = fre[0] if len(fre) > 1 else ''

    return tts

def tts_to_textGrid(tts, sword_flg=True):
    tg = TextGrid.TextGrid()
    tg.xmin = tts.xmin
    tg.xmax = tts.xmax
    st = tts.moras[0].get_st()
    ed = tts.moras[-1].get_ed()
    tg.append_sent(tts.text, st, ed)
    sword_st = tts.xmin
    for phr in tts.phrases:
        for wd in phr.words:
            if wd.moras[0].vow.startswith('sp'):
                if wd.moras[0].vow == 'sp1':
                    wd.moras[0].vow = 'sp'
                tg.append_word(wd.moras[0].vow, wd.moras[0].get_st(), wd.moras[0].get_ed())
                if sword_flg:
                    tg.append_sword('sp', sword_st, wd.moras[0].get_ed())
                    sword_st = wd.get_ed()
            kana = wd.get_kana()
            st = wd.get_st()
            ed = wd.get_ed()
            tg.append_word(kana, st, ed)
            for mr in wd.moras:
                st = mr.st
                if mr.hasCL:
                    ed = st + mr.CL_len
                    if mr.joinCL and tg.get_phoneme():
                        tg.get_phoneme(-1).text += ',<cl>'
                        tg.get_phoneme(-1).xmax += mr.CL_len
                    else:
                        tg.append_phoneme('<cl>', st, ed)
                    st = ed
                if mr.cons:
                    ed = st + mr.clen
                    if mr.join_C:
                        tg.get_phoneme(-1).text += f',{mr.cons}'
                        tg.get_phoneme(-1).xmax = ed
                    else:
                        tg.append_phoneme(mr.cons, st, ed)
                    st = ed
                if mr.hasJ:
                    ed = st + mr.J_len
                    if mr.joinJ:
                        tg.get_phoneme(-1).text += ',<j>'
                        tg.get_phoneme(-1).xmax = ed
                    else:
                        tg.append_phoneme('<j>', st, ed)
                    st = ed
                if mr.vow:
                    if mr.join_V:
                        tg.get_phoneme(-1).text += f',{mr.vow}'
                        tg.get_phoneme(-1).xmax += mr.vlen
                    else:
                        tg.append_phoneme(mr.vow, st, st + mr.vlen)

            if sword_flg:
                for swdix in range(len(wd.sword)+1):
                    swd, st, ed = wd.get_sword(swdix)
                    tg.append_sword(swd, st, ed)
                    sword_st = ed

        tg.get_word(-1).text += phr.bound_end.replace('.', '．').replace('?', '？').replace('!', '！')

    tg.addStEd()

    tg.addFrameNum(tts.fr_st, tts.fr_fps, tts.fr_fn)

    return tg


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument('kana')
    parser.add_argument('--tgrid', default='')
    parser.add_argument('--add_time', '-a', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--mode', '-m', default='0')
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


    if '0' in args.mode:
        if args.tgrid:
            tg = TextGrid.TextGrid(args.tgrid)
            kana_sent = utils.text.kana.KanaSent()
            kana = tg.get_jeitaKana()
            logging.debug(kana)
            kana_sent.set_kana(kana)
            logging.debug(str(kana_sent))
        else:
            tg = None

            '''
            for wd in tg.get_kanaSent().words:
                print(wd.bound_div, wd.bound_pau, wd.phs, wd.accup, wd.accdown, wd.acclevel, wd.bound_end)
                print(kana2roms(''.join(wd.phs)))
            '''

        tts = Tts()
        tts.from_kanaSent(kana_sent)
        if args.add_time:
            tts.set_time(tg)

        if tg:
            tts.xmin = tg.xmin
            tts.xmax = tg.xmax
            tts.text = tg.get_text()

        tts.lv_HtoXH()

        with open('tmp.tts', 'w') as ofs:
            print(tts, file=ofs)
        with open('tmp.tts', 'r') as ifs:
            tts2 = Tts(ifs)
        tg2 = tts_to_textGrid(tts)
        print(tg2)

    elif '1' in args.mode:
        tg = TextGrid.TextGrid(args.tgrid)
        tts = textGrid_to_Tts(tg)
        tts.div_all()
        tgo = tts_to_textGrid(tts)
        print(tgo)

    elif '2' in args.mode:
        tg = TextGrid.TextGrid(args.tgrid)
        tts = textGrid_to_Tts(tg)
        tgo = tts_to_textGrid(tts)
        tgo = tts_to_textGrid(tts)
        print(tgo)
