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
    'ヤ': ('j', 'a'),
    'ユ': ('j', 'u'),
    'ヨ': ('j', 'o'),
    'カ': ('k', 'a'),
    'キ': ('k', 'i'),
    'ク': ('k', 'u'),
    'ケ': ('k', 'e'),
    'コ': ('k', 'o'),
    'ガ': ('g', 'a'),
    'ギ': ('g', 'i'),
    'グ': ('g', 'u'),
    'ゲ': ('g', 'e'),
    'ゴ': ('g', 'o'),
    'サ': ('s', 'a'),
    'シ': ('s', 'i'),
    'ス': ('s', 'u'),
    'セ': ('s', 'e'),
    'ソ': ('s', 'o'),
    'ザ': ('z', 'a'),
    'ジ': ('z', 'i'),
    'ズ': ('z', 'u'),
    'ヅ': ('z', 'u'),
    'ゼ': ('z', 'e'),
    'ゾ': ('z', 'o'),
    'タ': ('t', 'a'),
    'チ': ('c', 'i'),
    'ツ': ('c', 'u'),
    'テ': ('t', 'e'),
    'ト': ('t', 'o'),
    'ダ': ('d', 'a'),
    'デ': ('d', 'e'),
    'ド': ('d', 'o'),
    'ナ': ('n', 'a'),
    'ニ': ('n', 'i'),
    'ヌ': ('n', 'u'),
    'ネ': ('n', 'e'),
    'ノ': ('n', 'o'),
    'ハ': ('h', 'a'),
    'ヒ': ('h', 'i'),
    'フ': ('h', 'u'),
    'ヘ': ('h', 'e'),
    'ホ': ('h', 'o'),
    'バ': ('b', 'a'),
    'ビ': ('b', 'i'),
    'ブ': ('b', 'u'),
    'ベ': ('b', 'e'),
    'ボ': ('b', 'o'),
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
    'イェ': ('j', 'e'),
    'キャ': ('kj', 'a'),
    'キュ': ('kj', 'u'),
    'キョ': ('kj', 'o'),
    'ギャ': ('gj', 'a'),
    'ギュ': ('gj', 'u'),
    'ギョ': ('gj', 'o'),
    'シャ': ('sj', 'a'),
    'シュ': ('sj', 'u'),
    'ショ': ('sj', 'o'),
    'シェ': ('sj', 'e'),
    'スィ': ('s_', 'i'),
    'ジャ': ('zj', 'a'),
    'ジュ': ('zj', 'u'),
    'ジョ': ('zj', 'o'),
    'ジェ': ('zj', 'e'),
    'ズィ': ('z_', 'i'),
    'チャ': ('cj', 'a'),
    'チュ': ('cj', 'u'),
    'チョ': ('cj', 'o'),
    'ティ': ('t_', 'i'),
    'トゥ': ('t', 'u'),
    'チェ': ('cj', 'e'),
    'ツァ': ('c', 'a'),
    'ツィ': ('c_', 'i'),
    'ツェ': ('c', 'e'),
    'ツォ': ('c', 'o'),
    'テュ': ('tj', 'u'),
    'ディ': ('d', 'i'),
    'ドゥ': ('d', 'u'),
    'デュ': ('dj', 'u'),
    'ニャ': ('nj', 'a'),
    'ニュ': ('nj', 'u'),
    'ニョ': ('nj', 'o'),
    'ニェ': ('nj', 'e'),
    'ヒャ': ('hj', 'a'),
    'ヒュ': ('hj', 'u'),
    'ヒョ': ('hj', 'o'),
    'ヒェ': ('hj', 'e'),
    'ファ': ('f', 'a'),
    'フィ': ('f', 'i'),
    'フェ': ('f', 'e'),
    'フォ': ('f', 'o'),
    'フュ': ('fj', 'u'),
    'ビャ': ('bj', 'a'),
    'ビュ': ('bj', 'u'),
    'ビョ': ('bj', 'o'),
    'ピャ': ('pj', 'a'),
    'ピュ': ('pj', 'u'),
    'ピョ': ('pj', 'o'),
    'ミャ': ('mj', 'a'),
    'ミュ': ('mj', 'u'),
    'ミョ': ('mj', 'o'),
    'ミェ': ('mj', 'e'),
    'リャ': ('rj', 'a'),
    'リュ': ('rj', 'u'),
    'リョ': ('rj', 'o'),
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

