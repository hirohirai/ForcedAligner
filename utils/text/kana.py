#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    kana.py
    Author: hirai
    Data: 2019/08/29
    無声は未対応
    SAMPAの異音も未対応
"""

import sys
import argparse
import logging
import re
import jaconv

# ログの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
logger.addHandler(stream_handler)
kana2ph1 = {'ア': ('a',),
'イ': ('i',),
'ウ': ('M',),
'エ': ('e',),
'オ': ('o',),
'ヲ': ('o',),
'カ': ('k','a'),
'キ': ('k','i'),
'ク': ('k','M'),
'ケ': ('k','e'),
'コ': ('k','o'),
'サ': ('s','a'),
'シ': ('S','i'),
'ス': ('s','M'),
'セ': ('s','e'),
'ソ': ('s','o'),
'タ': ('t','a'),
'チ': ('tS','i'),
'ツ': ('ts','M'),
'テ': ('t','e'),
'ト': ('t','o'),
'ナ': ('n','a'),
'ニ': ('J','i'),
'ヌ': ('n','M'),
'ネ': ('n','e'),
'ノ': ('n','o'),
'ハ': ('h','a'),
'ヒ': ('C','i'),
'フ': ('p\\','M'),
'ヘ': ('h','e'),
'ホ': ('h','o'),
'マ': ('m','a'),
'ミ': ('m','i'),
'ム': ('m','M'),
'メ': ('m','e'),
'モ': ('m','o'),
'ラ': ('4','a'),
'リ': ('4','i'),
'ル': ('4','M'),
'レ': ('4','e'),
'ロ': ('4','o'),
'ガ': ('g','a'),
'ギ': ('g','i'),
'グ': ('g','M'),
'ゲ': ('g','e'),
'ゴ': ('g','o'),
'ザ': ('dz','a'),
'ジ': ('dZ','i'),
'ズ': ('dz','M'),
'ゼ': ('dz','e'),
'ゾ': ('dz','o'),
'ダ': ('d','a'),
'ヅ': ('dz','M'),
'デ': ('d','e'),
'ド': ('d','o'),
'バ': ('b','a'),
'ビ': ('b','i'),
'ブ': ('b','M'),
'ベ': ('b','e'),
'ボ': ('b','o'),
'パ': ('p','a'),
'ピ': ('p','i'),
'プ': ('p','M'),
'ペ': ('p','e'),
'ポ': ('p','o'),
'ヴ': ('v','M'),
#'フ': ('f','M'),
'ン': ('N\\',),
'ヤ': ('j','a'),
'ユ': ('j','M'),
'ヨ': ('j','o'),
'ワ': ('w','a'),}

kana2ph2={'ディ': ('d','i'),
'ドゥ': ('d','M'),
'ティ': ('t','i'),
'トゥ': ('t','M'),
'ヴァ': ('v','a'),
'ヴィ': ('v','i'),
'ヴェ': ('v','e'),
'ヴォ': ('v','o'),
'カ゜': ('N','a'),
'キ゜': ('N','i'),
'ク゜': ('N','M'),
'ケ゜': ('N','e'),
'コ゜': ('N','o'),
'スィ': ('s','i'),
'ズィ': ('dz','i'),
'イェ': ('j','e'),
'キャ': ('kj','a'),
'キュ': ('kj','M'),
'キェ': ('kj','e'),
'キョ': ('kj','o'),
'シャ': ('S','a'),
'シュ': ('S','M'),
'シェ': ('S','e'),
'ショ': ('S','o'),
'チャ': ('tS','a'),
'チュ': ('tS','M'),
'チェ': ('tS','e'),
'チョ': ('tS','o'),
'ニャ': ('J','a'),
'ニュ': ('J','M'),
'ニェ': ('J','e'),
'ニョ': ('J','o'),
'ヒャ': ('C','a'),
'ヒュ': ('C','M'),
'ヒェ': ('C','e'),
'ヒョ': ('C','o'),
'ミャ': ('mj','a'),
'ミュ': ('mj','M'),
'ミェ': ('mj','e'),
'ミョ': ('mj','o'),
'リャ': ('4j','a'),
'リュ': ('4j','M'),
'リェ': ('4j','e'),
'リョ': ('4j','o'),
'ギャ': ('gj','a'),
'ギュ': ('gj','M'),
'ギェ': ('gj','e'),
'ギョ': ('gj','o'),
'ジャ': ('dZ','a'),
'ジュ': ('dZ','M'),
'ジェ': ('dZ','e'),
'ジョ': ('dZ','o'),
'デャ': ('dj','a'),
'デュ': ('dj','M'),
'デョ': ('dj','o'),
'ビャ': ('bj','a'),
'ビュ': ('bj','M'),
'ビェ': ('bj','e'),
'ビョ': ('bj','o'),
'ピャ': ('pj','a'),
'ピュ': ('pj','M'),
'ピェ': ('pj','e'),
'ピョ': ('pj','o'),
'テャ': ('tj','a'),
'テュ': ('tj','M'),
'テョ': ('tj','o'),
'ヴャ': ('vj','a'),
'ヴュ': ('vj','M'),
'ヴョ': ('vj','o'),
'フャ': ('P\\j','a'),
'フュ': ('P\\j','M'),
'フョ': ('P\\j','o'),
'ウィ': ('w','i'),
'ウェ': ('w','e'),
'ウォ': ('w','o'),
'クァ': ('k_w','a'),
'クィ': ('k_w','i'),
'クェ': ('k_w','e'),
'クォ': ('k_w','o'),
'スァ': ('s_w','a'),
'スェ': ('s_w','e'),
'スォ': ('s_w','o'),
'ツァ': ('ts','a'),
'ツィ': ('ts','i'),
'ツェ': ('ts','e'),
'ツォ': ('ts','o'),
'ヌァ': ('n_w','a'),
'ヌィ': ('n_w','i'),
'ヌェ': ('n_w','e'),
'ヌォ': ('n_w','o'),
'ファ': ('p\\','a'),
'フィ': ('p\\','i'),
'フェ': ('p\\','e'),
'フォ': ('p\\','o'),
'ムァ': ('m_w','a'),
'ムィ': ('m_w','i'),
'ムェ': ('m_w','e'),
'ムォ': ('m_w','o'),
'ルァ': ('4_w','a'),
'ルィ': ('4_w','i'),
'ルェ': ('4_w','e'),
'ルォ': ('4_w','o'),
'グァ': ('g_w','a'),
'グィ': ('g_w','i'),
'グェ': ('g_w','e'),
'グォ': ('g_w','o'),
'ズァ': ('dz_w','a'),
#'ズィ': ('dz_w','i'),
'ズェ': ('dz_w','e'),
'ズォ': ('dz_w','o'),
'ブァ': ('b_w','a'),
'ブィ': ('b_w','i'),
'ブェ': ('b_w','e'),
'ブォ': ('b_w','o'),
'プァ': ('p_w','a'),
'プィ': ('p_w','i'),
'プェ': ('p_w','e'),
'プォ': ('p_w','o'),}


kana2ph3 = {'ディェ': ('dj','e'),
'ティェ': ('tj','e'),
'ヴィェ': ('vj','e'),
'キ゜ャ': ('Nj','a'),
'キ゜ュ': ('Nj','M'),
'キ゜ェ': ('Nj','e'),
'キ゜ョ': ('Nj','o'),
'フィェ': ('P\\j','e'),
'スゥィ': ('s_w','i'),
'ドゥァ': ('d_w','a'),
'ドゥィ': ('d_w','i'),
'ドゥェ': ('d_w','e'),
'ドゥォ': ('d_w','o'),
'トゥァ': ('t_w','a'),
'トゥィ': ('t_w','i'),
'トゥェ': ('t_w','e'),
'トゥォ': ('t_w','o'),
'ヴゥァ': ('v_w','a'),
'ヴゥィ': ('v_w','i'),
'ヴゥェ': ('v_w','e'),
'ヴゥォ': ('v_w','o'),
'ク゜ェ': ('N_w','e'),
'ク゜ォ': ('N_w','o'),
'ク゜ァ': ('N_w','a'),
'ク゜ィ': ('N_w','i'),}


def kana2sampa(ka):
    if len(ka) == 3:
        return kana2ph3[ka]
    elif len(ka) == 2:
        return kana2ph2[ka]
    else:
        return kana2ph1[ka]


cnvNorm = str.maketrans({"'": "’", ',': '’', '/': '／', '_': '＿', '|': '｜', '!': '！', '?': '？', '０': '0', '１': '1', '２': '2'})

div_wd_bnd = re.compile(r"(／＿|／＿\d|／＿+|／|＿\d|＿+|｜|　|，)")

class Word:
    def __init__(self):
        self.phs = []
        self.sword = []
        self.sword_kana = []
        self.accup = 1
        self.accdown = 0
        self.acclevel = 1
        self.bound_div = '／'  # ／　｜　，　'　'
        self.bound_pau = ''  # ＿　＿0 ＿１ ＿２
        self.bound_end = ''  # ．？ ！ ！？

    def set_bound_f(self, bnd):
        if bnd[0] in '／｜，　':
            self.bound_div = bnd[0]
            bnd = bnd[1:]
        if len(bnd) > 0:
            self.bound_pau = bnd

    def set_bound_b(self, bnd):
        self.bound_end = bnd

    def set_jeitaKana(self, ka):
        ka = ka.translate(cnvNorm)
        self.phs = []
        while len(ka)>0:
            if ka[:3] in kana2ph3.keys():
                self.phs.append(ka[:3])
                ka = ka[3:]
            elif ka[:2] in kana2ph2.keys():
                self.phs.append(ka[:2])
                ka = ka[2:]
            elif ka[:1] in kana2ph1.keys():
                self.phs.append(ka[:1])
                ka = ka[1:]
            elif ka[0] == "’":
                self.accdown = len(self.phs)
                if self.accdown != 1:
                    self.accup = 1
                else:
                    self.accup = 0
                ka = ka[1:]
                if ka[:1] in '012':
                    self.acclevel = int(ka[0])
                    ka = ka[1:]
            elif ka[0] in 'ー-−―':
                self.phs.append('ー')
                ka = ka[1:]
            elif ka[0] == 'ッ':
                self.phs.append(ka[0])
                ka = ka[1:]
            elif ka[0] in '／｜，　':
                self.set_bound_f(ka[:1])
                ka = ka[1:]
            elif ka[0] == '＿':
                if len(ka)>1 and ka[1] in '012':
                    self.set_bound_f(ka[:2])
                    ka = ka[2:]
                else:
                    self.set_bound_f(ka[:1])
                    ka = ka[1:]
            elif len(ka) == 1 and ka[0] in '．？！':
                self.set_bound_b(ka[:1])
                ka = ka[1:]
            elif len(ka) == 2 and ka == '！？':
                self.set_bound_b(ka)
                ka = ka[2:]
            else:
                ka = ka[1:]
            '''
            elif ka[0] in '012':
                if len(ka) > 1 and ka[1] in '．／｜＿？！　，':
                    self.acclevel = int(ka[0])
                    if ka[1] in '／｜，　＿':
                        self.set_bound_f(ka[1])
                    else:
                        self.set_bound_b(ka[1])
                    ka = ka[2:]
                elif len(ka) == 3 and ka[1:] == '！？':
                    self.acclevel = int(ka[0])
                    self.set_bound_b(ka[1:])
                    ka = ka[3:]
                else:
                    ka = ka[1:]
            '''


        logging.debug(f'DIV: {self.bound_div}')
        logging.debug(f'PAU: {self.bound_pau}')
        logging.debug(f'END: {self.bound_end}')


    def add_sword(self, swrs, spo):
        pp = 0
        c_swrd = swrs[spo].split(':')[0] if spo < len(swrs) else ''
        for phix, ph in enumerate(self.phs, 1):
            if c_swrd[pp:].startswith(ph):
                pp += len(ph)
                if len(c_swrd) == pp:
                    spo += 1
                    pp = 0
                    c_swrd = swrs[spo].split(':')[0] if spo < len(swrs) else ''
                    if phix < len(self.phs):
                        self.sword.append(phix)
                    self.sword_kana.append(swrs[spo - 1])
            else:
                logger.error(f'sword NEQ: {swrs[spo]} {pp} {ph}')

        return spo


    def get_jeitaKana(self, sep=''):
        outs = []
        st = 0
        outs.append(self.bound_div)
        outs.append(self.bound_pau)
        if self.accdown > 0:
            for ix in range(st, self.accdown):
                outs.append(self.phs[ix])
            outs.append('’')
            if self.acclevel != 1:
                outs.append(f'{self.acclevel}')
            st = self.accdown
        for ix in range(st, len(self.phs)):
            outs.append(self.phs[ix])
        outs.append(self.bound_end)
        return outs

    def get_kana(self, sep=''):
        outs = []
        for ix in range(0, len(self.phs)):
            outs.append(self.phs[ix])
        return sep.join(outs)

    def get_sampa(self):
        outs = []
        if self.bound_div == '／':  # アクセント区切り
            outs.append('|')
        elif self.bound_div == '，':  # 並列の区切り
            outs.append('|_/')
        elif self.bound_div == '　':  # 副次アクセント区切り
            outs.append('-')
        elif self.bound_div == '｜':  # フレーズ区切り
            outs.append('||')

        if self.bound_pau == '＿0':  # 数字はちょっと違うがこれで　単ポーズ
            outs.append('(.)')
        elif self.bound_pau == '＿1' or self.bound_pau == '＿':  # 少し短めのポーズ
            outs.append('(..)')
        elif self.bound_pau == '＿2':  # 読点の長いポーズ
            outs.append('(...)')

        if len(self.bound_div) == 0 and len(self.bound_pau) == 0:
            outs.append('|')

        if len(self.phs)>0:
            st = 0
            QQNUM = 0
            if self.accup > 0:
                for ix in range(st, self.accup):
                    if self.phs[ix] == 'ー':
                        if QQNUM > 0:
                            QQNUM += 1
                        else:
                            outs.append(':')
                    elif self.phs[ix] == 'ッ':
                        QQNUM += 1
                    else:
                        if QQNUM > 0:
                            smp = kana2sampa(self.phs[ix])
                            if len(smp) == 1:
                                smp = ('q',)
                            for ii in range(QQNUM):
                                outs.append(smp[0])
                            QQNUM = 0
                        smp = kana2sampa(self.phs[ix])
                        outs += smp
                outs.append('^')
                st = self.accup

            if self.accdown > 0:
                for ix in range(st, self.accdown):
                    if self.phs[ix] == 'ー':
                        if QQNUM>0:
                            QQNUM += 1
                        else:
                            outs.append(':')
                    elif self.phs[ix] == 'ッ':
                        QQNUM += 1
                    else:
                        if QQNUM > 0:
                            smp = kana2sampa(self.phs[ix])
                            if len(smp)==1:
                                smp = ('q',)
                            for ii in range(QQNUM):
                                outs.append(smp[0])
                            QQNUM = 0
                        smp = kana2sampa(self.phs[ix])
                        outs += smp
                if self.acclevel == 0:
                    outs.append('!0')
                elif self.acclevel == 2:
                    outs.append('!2')
                else:
                    outs.append('!')
                st = self.accdown
            for ix in range(st, len(self.phs)):
                if self.phs[ix] == 'ー':
                    if QQNUM > 0:
                        QQNUM += 1
                    else:
                        outs.append(':')
                elif self.phs[ix] == 'ッ':
                    QQNUM += 1
                else:
                    if QQNUM > 0:
                        smp = kana2sampa(self.phs[ix])
                        if len(smp) == 1:
                            smp = ('q',)
                        for ii in range(QQNUM):
                            outs.append(smp[0])
                        QQNUM = 0
                    smp = kana2sampa(self.phs[ix])
                    outs += smp

            if self.accdown == 0 and self.acclevel != 1:
                outs.append(str(self.acclevel))

        if self.bound_end == '．':  # 通常の終端
            outs.append("<F>||")
        elif self.bound_end == '？':  # 疑問の終端
            outs.append("<R>||")
        elif self.bound_end == '!':  # 断定の終端
            outs.append("<F>_F||")
        elif self.bound_end == '！？':  # 上昇調の「ね」
            outs.append("<R>_F||")
        return outs

    def __str__(self):
        outs = f'ACC:{self.accup},{self.accdown},{self.acclevel} BOUND:{self.bound_div},{self.bound_pau},{self.bound_end}\n'
        for ph in self.phs:
            outs += f'{ph} '
        return outs


class KanaSent:
    def __init__(self):
        self.words = []

    def set_bound(self, bnd=None):
        if len(self.words) > 0:
            if bnd:
                self.words[-1].set_bound_b(bnd)
            else:
                if self.words[-1].bound_end == '':
                    self.words[-1].set_bound_b('．')

    def add_word_textGrid(self, ka):
        if len(ka)>0 and ka[-1] == '’':
            ka = ka[:-1]

        if len(ka) == 0 or ka == '#':
            return
        stposi = len(self.words)

        if ka.startswith('sp') or ka.endswith('sp'):
            if len(self.words) > 0:
                self.words.append(Word())
                self.words[-1].set_bound_f('｜＿')
        else:
            self.add_jeitaKana(ka, end_flg=False)

        if stposi == 0:
            if self.words[0].bound_div == '／' and ka[0] != '／':
                self.words[0].bound_div = ''
        elif len(self.words[stposi-1].phs)==0 and ka[0] != '／':
            self.words[stposi].bound_div = ''


    def add_jeitaKana(self, jkana, end_flg=False):
        jkana_d = div_wd_bnd.sub(r" \1", jkana.strip()).strip().split(' ')
        for kk in jkana_d:
            wrd = Word()
            logging.debug("add_jk: " + kk)
            wrd.set_jeitaKana(kk)
            self.words.append(wrd)
        if end_flg:
            self.set_bound()

    def add_sword(self, swrds):
        spo = 0
        for wd in self.words:
            spo = wd.add_sword(swrds, spo)
        if spo != len(swrds):
            logging.error("S UNIT ERROR")

    def get_jeitaKana(self):
        rets = []
        for wd in self.words:
            rets += wd.get_jeitaKana()
        return rets

    def get_juliusKana(self, sep=''):
        rets = []
        for wd in self.words:
            if wd.bound_pau:
                rets.append(' sp')
            rets.append(jaconv.kata2hira(wd.get_kana()))

        return sep.join(rets)

    def get_sampa(self):
        rets = []
        for wd in self.words:
            rets += wd.get_sampa()
        return rets

    def get_input_feature(self):
        return self.get_sampa()

    def set_kana(self, kana):
        kwrd = ''
        bound = ''
        for kk in kana:
            if kk =='':
                continue
            if kk in '／｜，　＿':
                if kwrd:
                    self.add_jeitaKana(bound + kwrd)
                    kwrd = ''
                    bound = kk
                else:
                    bound += kk
            else:
                kwrd += kk
        if kwrd:
            self.add_jeitaKana(bound + kwrd)

    def __str__(self):
        outs = ''
        for wrd in self.words:
            outs += f'{wrd}\n'
        return outs

def dumpallsampa(args):
    for kk in kana2ph1.keys():
        for dd in kana2ph1[kk]:
            dd = "'"+dd +"',"
            print(dd)
    for kk in kana2ph2.keys():
        for dd in kana2ph2[kk]:
            dd = "'"+dd +"',"
            print(dd)
    for kk in kana2ph3.keys():
        for dd in kana2ph3[kk]:
            dd = "'"+dd +"',"
            print(dd)

def main(args):
    kana = KanaSent()
    for kk in args.input:
        kk = kk.strip()
        kana.add_word_textGrid(kk)
    kana.set_bound()

    print(args.sep.join(kana.get_input_feature()))

def main2(args):

    for kk in args.input:
        kk = kk.strip()
        kana = KanaSent()
        kana.set_jeitaKana(kk)

        print(args.sep.join(kana.get_input_feature()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sep", default='')
    parser.add_argument("-i", "--input", type=argparse.FileType("r"), default="-")

    args = parser.parse_args()

    main2(args)
    #dumpallsampa(args)