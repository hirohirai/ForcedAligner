#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""

import sys, io
import argparse
import logging


kana2rom_v_tbl = {
    'ア': ('','a'),
    'イ': ('','i'),
    'ウ': ('','u'),
    'エ': ('','e'),
    'オ': ('','o'),
    'ン': ('','N'),
    'ヲ': ('', 'o'),
    }

def kana2rom_v(kn):
    if kn in kana2rom_v_tbl:
        return kana2rom_v_tbl[kn]
    return None

kana2rom_c1_tbl = {
    'ヤ': ('y', 'a'),
    'ユ': ('y', 'u'),
    'ヨ': ('y', 'o'),
    'カ': ('k', 'a'),
    'キ': ('kj', 'i'),
    'ク': ('k', 'u'),
    'ケ': ('k', 'e'),
    'コ': ('k', 'o'),
    'ガ': ('g', 'a'),
    'ギ': ('gj', 'i'),
    'グ': ('g', 'u'),
    'ゲ': ('g', 'e'),
    'ゴ': ('g', 'o'),
    'サ': ('s', 'a'),
    'シ': ('sj', 'i'),
    'ス': ('s', 'u'),
    'セ': ('s', 'e'),
    'ソ': ('s', 'o'),
    'ザ': ('z', 'a'),
    'ジ': ('zj', 'i'),
    'ズ': ('z', 'u'),
    'ヅ': ('z', 'u'),
    'ゼ': ('z', 'e'),
    'ゾ': ('z', 'o'),
    'タ': ('t', 'a'),
    'チ': ('cj', 'i'),
    'ツ': ('c', 'u'),
    'テ': ('t', 'e'),
    'ト': ('t', 'o'),
    'ダ': ('d', 'a'),
    'デ': ('d', 'e'),
    'ド': ('d', 'o'),
    'ナ': ('n', 'a'),
    'ニ': ('nj', 'i'),
    'ヌ': ('n', 'u'),
    'ネ': ('n', 'e'),
    'ノ': ('n', 'o'),
    'ハ': ('h', 'a'),
    'ヒ': ('hj', 'i'),
    'フ': ('F', 'u'),
    'ヘ': ('h', 'e'),
    'ホ': ('h', 'o'),
    'バ': ('b', 'a'),
    'ビ': ('b', 'i'),
    'ブ': ('b', 'u'),
    'ベ': ('b', 'e'),
    'ボ': ('b', 'o'),
    'ヴ': ('v', 'u'),
    'パ': ('p', 'a'),
    'ピ': ('p', 'i'),
    'プ': ('p', 'u'),
    'ペ': ('p', 'e'),
    'ポ': ('p', 'o'),
    'マ': ('m', 'a'),
    'ミ': ('m', 'i'),
    'ム': ('m', 'u'),
    'メ': ('m', 'e'),
    'モ': ('m', 'o'),
    'ラ': ('r', 'a'),
    'リ': ('r', 'i'),
    'ル': ('r', 'u'),
    'レ': ('r', 'e'),
    'ロ': ('r', 'o'),
    'ワ': ('w', 'a'),
}

kana2rom_c2_tbl = {
    'イェ': ('y', 'e'),
    'キャ': ('ky', 'a'),
    'キュ': ('ky', 'u'),
    'キョ': ('ky', 'o'),
    'クヮ': ('kw', 'a'),
    'ギャ': ('gy', 'a'),
    'ギュ': ('gy', 'u'),
    'ギョ': ('gy', 'o'),
    'グヮ': ('gw', 'a'),
    'シャ': ('sy', 'a'),
    'シュ': ('sy', 'u'),
    'ショ': ('sy', 'o'),
    'シェ': ('sy', 'e'),
    'スィ': ('s', 'i'),
    'ジャ': ('zy', 'a'),
    'ジュ': ('zy', 'u'),
    'ジョ': ('zy', 'o'),
    'ジェ': ('zy', 'e'),
    'ズィ': ('z', 'i'),
    'チャ': ('cy', 'a'),
    'チュ': ('cy', 'u'),
    'チョ': ('cy', 'o'),
    'ティ': ('t', 'i'),
    'トゥ': ('t', 'u'),
    'チェ': ('cy', 'e'),
    'ツァ': ('c', 'a'),
    'ツィ': ('c', 'i'),
    'ツェ': ('c', 'e'),
    'ツォ': ('c', 'o'),
    'テュ': ('ty', 'u'),
    'ディ': ('d', 'i'),
    'ドゥ': ('d', 'u'),
    'デュ': ('dy', 'u'),
    'ニャ': ('ny', 'a'),
    'ニュ': ('ny', 'u'),
    'ニョ': ('ny', 'o'),
    'ニェ': ('ny', 'e'),
    'ヒャ': ('hy', 'a'),
    'ヒュ': ('hy', 'u'),
    'ヒョ': ('hy', 'o'),
    'ヒェ': ('hy', 'e'),
    'ファ': ('F', 'a'),
    'フィ': ('F', 'i'),
    'フェ': ('F', 'e'),
    'フォ': ('F', 'o'),
    'フュ': ('Fy', 'u'),
    'ビャ': ('by', 'a'),
    'ビュ': ('by', 'u'),
    'ビョ': ('by', 'o'),
    'ヴァ': ('v', 'a'),
    'ヴィ': ('v', 'i'),
    'ヴェ': ('v', 'e'),
    'ヴォ': ('v', 'o'),
    'ピャ': ('py', 'a'),
    'ピュ': ('py', 'u'),
    'ピョ': ('py', 'o'),
    'ミャ': ('my', 'a'),
    'ミュ': ('my', 'u'),
    'ミョ': ('my', 'o'),
    'ミェ': ('my', 'e'),
    'リャ': ('ry', 'a'),
    'リュ': ('ry', 'u'),
    'リョ': ('ry', 'o'),
    'ウィ': ('w', 'i'),
    'ウェ': ('w', 'e'),
    'ウォ': ('w', 'o'),
}



def kana2rom_c(kn):
    if kn in kana2rom_c2_tbl:
        return kana2rom_c2_tbl[kn]
    if kn in kana2rom_c1_tbl:
        return kana2rom_c1_tbl[kn]
    return None

def kana2rom_vc(kn):
    if kn in kana2rom_c2_tbl:
        return kana2rom_c2_tbl[kn]
    if kn in kana2rom_c1_tbl:
        return kana2rom_c1_tbl[kn]
    if kn in kana2rom_v_tbl:
        return kana2rom_v_tbl[kn]
    return None

def kana2roms(kn):
    roms = []
    ix = 0
    while ix < len(kn)-1:
        if kn[ix:ix+2] in kana2rom_c2_tbl:
            roms.append(kana2rom_c2_tbl[kn[ix:ix + 2]])
            ix += 1
        elif kn[ix] in kana2rom_c1_tbl:
            roms.append(kana2rom_c1_tbl[kn[ix]])
        elif kn[ix] in kana2rom_v_tbl:
            roms.append(kana2rom_v_tbl[kn[ix]])
        elif kn[ix] == 'ー':
            roms.append(('', 'H'))
        elif kn[ix] == 'ッ':
            roms.append(('Q', ''))

        ix += 1

    if ix < len(kn):
        if kn[ix] in kana2rom_c1_tbl:
            roms.append(kana2rom_c1_tbl[kn[ix]])
        elif kn[ix] in kana2rom_v_tbl:
            roms.append(kana2rom_v_tbl[kn[ix]])
        elif kn[ix] == 'ー':
            roms.append(('', 'H'))
    return roms

