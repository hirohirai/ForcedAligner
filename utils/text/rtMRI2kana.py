#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""

import sys, io
import argparse
import logging


rom2kana_v_tbl = {
    'a':'ア',
    'i':'イ',
    'u':'ウ',
    'e':'エ',
    'o':'オ',
    'N': 'ン',
    'H': 'ー',
    'Q': 'ッ',
    }

def rom2kana_v(rom):
    if rom[-1] == 'H':
        return rom2kana_v_tbl['H']
    if rom in 'AIUEO':
        rom = rom.lower()
    if rom in rom2kana_v_tbl:
        return rom2kana_v_tbl[rom]
    return ''

rom2kana_c_tbl = {
    ('j', 'a'):'ヤ',
    ('j', 'u'):'ユ',
    ('j', 'o'):'ヨ',
    ('k', 'a'):'カ',
    ('k', 'i'):'キ',
    ('k', 'u'):'ク',
    ('k', 'e'):'ケ',
    ('k', 'o'):'コ',
    ('g', 'a'):'ガ',
    ('g', 'i'):'ギ',
    ('g', 'u'):'グ',
    ('g', 'e'):'ゲ',
    ('g', 'o'):'ゴ',
    ('s', 'a'):'サ',
    ('s', 'i'):'シ',
    ('s', 'u'):'ス',
    ('s', 'e'):'セ',
    ('s', 'o'):'ソ',
    ('z', 'a'):'ザ',
    ('z', 'i'):'ジ',
    ('z', 'u'):'ズ',
    ('z', 'e'):'ゼ',
    ('z', 'o'):'ゾ',
    ('t', 'a'):'タ',
    ('c', 'i'):'チ',
    ('c', 'u'):'ツ',
    ('t', 'e'):'テ',
    ('t', 'o'):'ト',
    ('d', 'a'):'ダ',
    ('d', 'e'):'デ',
    ('d', 'o'):'ド',
    ('n', 'a'):'ナ',
    ('n', 'i'):'ニ',
    ('n', 'u'):'ヌ',
    ('n', 'e'):'ネ',
    ('n', 'o'):'ノ',
    ('h', 'a'):'ハ',
    ('h', 'i'):'ヒ',
    ('h', 'u'):'フ',
    ('h', 'e'):'ヘ',
    ('h', 'o'):'ホ',
    ('b', 'a'):'バ',
    ('b', 'i'):'ビ',
    ('b', 'u'):'ブ',
    ('b', 'e'):'ベ',
    ('b', 'o'):'ボ',
    ('p', 'a'):'パ',
    ('p', 'i'):'ピ',
    ('p', 'u'):'プ',
    ('p', 'e'):'ペ',
    ('p', 'o'):'ポ',
    ('m', 'a'):'マ',
    ('m', 'i'):'ミ',
    ('m', 'u'):'ム',
    ('m', 'e'):'メ',
    ('m', 'o'):'モ',
    ('r', 'a'):'ラ',
    ('r', 'i'):'リ',
    ('r', 'u'):'ル',
    ('r', 'e'):'レ',
    ('r', 'o'):'ロ',
    ('w', 'a'):'ワ',
    #('w', 'o'):'ヲ',
     ('j', 'e'):'イェ',
     ('kj', 'a'):'キャ',
     ('kj', 'u'):'キュ',
     ('kj', 'o'):'キョ',
     ('gj', 'a'):'ギャ',
     ('gj', 'u'):'ギュ',
     ('gj', 'o'):'ギョ',
     ('sj', 'a'):'シャ',
     ('sj', 'u'):'シュ',
     ('sj', 'o'):'ショ',
     ('sj', 'e'):'シェ',
     ('s_', 'i'):'スィ',
     ('zj', 'a'):'ジャ',
     ('zj', 'u'):'ジュ',
     ('zj', 'o'):'ジョ',
     ('zj', 'e'):'ジェ',
     ('z_', 'i'):'ズィ',
     ('cj', 'a'):'チャ',
     ('cj', 'u'):'チュ',
     ('cj', 'o'):'チョ',
     ('t_', 'i'):'ティ',
     ('t', 'u'):'トゥ',
     ('cj', 'e'):'チェ',
     ('c', 'a'):'ツァ',
     ('c_', 'i'):'ツィ',
     ('c', 'e'):'ツェ',
     ('c', 'o'):'ツォ',
     ('tj', 'u'):'テュ',
     ('d', 'i'):'ディ',
     ('d', 'u'):'ドゥ',
     ('dj', 'u'):'デュ',
     ('nj', 'a'):'ニャ',
     ('nj', 'u'):'ニュ',
     ('nj', 'o'):'ニョ',
     ('nj', 'e'):'ニェ',
     ('hj', 'a'):'ヒャ',
     ('hj', 'u'):'ヒュ',
     ('hj', 'o'):'ヒョ',
     ('hj', 'e'):'ヒェ',
     ('f', 'a'):'ファ',
     ('f', 'i'):'フィ',
     ('f', 'e'):'フェ',
     ('f', 'o'):'フォ',
     ('fj', 'u'):'フュ',
     ('bj', 'a'):'ビャ',
     ('bj', 'u'):'ビュ',
     ('bj', 'o'):'ビョ',
     ('pj', 'a'):'ピャ',
     ('pj', 'u'):'ピュ',
     ('pj', 'o'):'ピョ',
     ('mj', 'a'):'ミャ',
     ('mj', 'u'):'ミュ',
     ('mj', 'o'):'ミョ',
     ('mj', 'e'):'ミェ',
     ('rj', 'a'):'リャ',
     ('rj', 'u'):'リュ',
     ('rj', 'o'):'リョ',
     ('w', 'i'):'ウィ',
    ('w', 'e'):'ウェ',
    ('w', 'o'):'ウォ',
    ('Q', ''):'ッ',
}

def rom2kana_c(cc, vv):
    if vv in 'AIUEO':
        vv = vv.lower()
    if len(cc) > 0:
        if (cc, vv) in rom2kana_c_tbl:
            return rom2kana_c_tbl[(cc,vv)]
    else:
        return rom2kana_v(vv)
    return ''

def roms2kanas(roms):
    outs = []
    while roms:
        if roms[0] == 'sp':
            outs.append(' sp')
        elif rom2kana_v(roms[0]):
            outs.append(rom2kana_v(roms[0]))
        elif len(roms) > 1:
            okana = rom2kana_c(roms[0], roms[1])
            outs.append(okana)
            roms = roms[1:]
        roms = roms[1:]
    return outs

if __name__ == "__main__":
    for ll in sys.stdin:
        print(''.join(roms2kanas(ll.strip().split())))
