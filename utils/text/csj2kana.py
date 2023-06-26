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
    ('y', 'a'):'ヤ',
    ('y', 'u'):'ユ',
    ('y', 'o'):'ヨ',
    ('k', 'a'):'カ',
    ('kj', 'i'):'キ',
    ('k', 'u'):'ク',
    ('k', 'e'):'ケ',
    ('k', 'o'):'コ',
    ('g', 'a'):'ガ',
    ('gj', 'i'):'ギ',
    ('g', 'u'):'グ',
    ('g', 'e'):'ゲ',
    ('g', 'o'):'ゴ',
    ('s', 'a'):'サ',
    ('sj', 'i'):'シ',
    ('s', 'u'):'ス',
    ('s', 'e'):'セ',
    ('s', 'o'):'ソ',
    ('z', 'a'):'ザ',
    ('zj', 'i'):'ジ',
    ('z', 'u'):'ズ',
    ('z', 'e'):'ゼ',
    ('z', 'o'):'ゾ',
    ('t', 'a'):'タ',
    ('cj', 'i'):'チ',
    ('c', 'u'):'ツ',
    ('t', 'e'):'テ',
    ('t', 'o'):'ト',
    ('d', 'a'):'ダ',
    ('d', 'e'):'デ',
    ('d', 'o'):'ド',
    ('n', 'a'):'ナ',
    ('nj', 'i'):'ニ',
    ('n', 'u'):'ヌ',
    ('n', 'e'):'ネ',
    ('n', 'o'):'ノ',
    ('h', 'a'):'ハ',
    ('hj', 'i'):'ヒ',
    ('F', 'u'):'フ',
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
     ('ky', 'a'):'キャ',
     ('ky', 'u'):'キュ',
    ('ky', 'o'):'キョ',
    ('kw', 'a'):'クヮ',
     ('gy', 'a'):'ギャ',
     ('gy', 'u'):'ギュ',
    ('gy', 'o'): 'ギョ',
    ('gw', 'a'): 'グヮ',
     ('sy', 'a'):'シャ',
     ('sy', 'u'):'シュ',
     ('sy', 'o'):'ショ',
     ('sy', 'e'):'シェ',
     ('s', 'i'):'スィ',
     ('zy', 'a'):'ジャ',
     ('zy', 'u'):'ジュ',
     ('zy', 'o'):'ジョ',
     ('zy', 'e'):'ジェ',
     ('z', 'i'):'ズィ',
     ('cy', 'a'):'チャ',
     ('cy', 'u'):'チュ',
     ('cy', 'o'):'チョ',
     ('t', 'i'):'ティ',
     ('t', 'u'):'トゥ',
     ('cy', 'e'):'チェ',
     ('c', 'a'):'ツァ',
     ('c', 'i'):'ツィ',
     ('c', 'e'):'ツェ',
     ('c', 'o'):'ツォ',
     ('ty', 'u'):'テュ',
     ('d', 'i'):'ディ',
     ('d', 'u'):'ドゥ',
     ('dy', 'u'):'デュ',
     ('ny', 'a'):'ニャ',
     ('ny', 'u'):'ニュ',
     ('ny', 'o'):'ニョ',
     ('ny', 'e'):'ニェ',
     ('hy', 'a'):'ヒャ',
     ('hy', 'u'):'ヒュ',
     ('hy', 'o'):'ヒョ',
     ('hy', 'e'):'ヒェ',
     ('F', 'a'):'ファ',
     ('F', 'i'):'フィ',
     ('F', 'e'):'フェ',
     ('F', 'o'):'フォ',
     ('Fy', 'u'):'フュ',
     ('by', 'a'):'ビャ',
     ('by', 'u'):'ビュ',
    ('by', 'o'): 'ビョ',
    ('v', 'a'): 'ヴァ',
    ('v', 'i'): 'ヴィ',
    ('v', 'u'): 'ヴ',
    ('v', 'e'): 'ヴェ',
    ('v', 'o'): 'ヴォ',
     ('py', 'a'):'ピャ',
     ('py', 'u'):'ピュ',
     ('py', 'o'):'ピョ',
     ('my', 'a'):'ミャ',
     ('my', 'u'):'ミュ',
     ('my', 'o'):'ミョ',
     ('my', 'e'):'ミェ',
     ('ry', 'a'):'リャ',
     ('ry', 'u'):'リュ',
     ('ry', 'o'):'リョ',
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
