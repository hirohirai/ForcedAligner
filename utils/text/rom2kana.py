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
    'A':'ア',
    'I':'イ',
    'U':'ウ',
    'E':'エ',
    'O':'オ',
    'N':'ン',
    'wo':'ヲ',
    'he':'ヱ',
    }

def rom2kana_v(rom):
    if rom in rom2kana_v_tbl:
        return rom2kana_v_tbl[rom]
    elif len(rom)>0 and rom[-1] == ':':
        return 'ー'
    return ''

rom2kana_c_tbl = {
     ('k','a'):'カ',
     ('k','i'):'キ',
     ('k','u'):'ク',
     ('k','e'):'ケ',
     ('k','o'):'コ',
     ('s','a'):'サ',
     ('sh','i'):'シ',
     ('s','i'):'スィ',
     ('s','u'):'ス',
     ('s','e'):'セ',
     ('s','o'):'ソ',
     ('t','a'):'タ',
     ('ch','i'):'チ',
     ('t','i'):'ティ',
     ('ts','u'):'ツ',
     ('t','u'):'トゥ',
     ('t','e'):'テ',
     ('t','o'):'ト',
     ('n','a'):'ナ',
     ('n','i'):'ニ',
     ('n','u'):'ヌ',
     ('n','e'):'ネ',
     ('n','o'):'ノ',
     ('h','a'):'ハ',
     ('h','i'):'ヒ',
     ('h','u'):'フ',
     ('h','e'):'ヘ',
     ('h','o'):'ホ',
     ('m','a'):'マ',
     ('m','i'):'ミ',
     ('m','u'):'ム',
     ('m','e'):'メ',
     ('m','o'):'モ',
     ('r','a'):'ラ',
     ('r','i'):'リ',
     ('r','u'):'ル',
     ('r','e'):'レ',
     ('r','o'):'ロ',
     ('g','a'):'ガ',
     ('g','i'):'ギ',
     ('g','u'):'グ',
     ('g','e'):'ゲ',
     ('g','o'):'ゴ',
     ('z','a'):'ザ',
     ('j','i'):'ジ',
     ('z','i'):'ズィ',
     ('z','u'):'ズ',
     ('z','e'):'ゼ',
     ('z','o'):'ゾ',
     ('d','a'):'ダ',
     #('z','u'):'ヅ',
     ('d','e'):'デ',
     ('d','o'):'ド',
     ('b','a'):'バ',
     ('b','i'):'ビ',
     ('b','u'):'ブ',
     ('b','e'):'ベ',
     ('b','o'):'ボ',
     ('p','a'):'パ',
     ('p','i'):'ピ',
     ('p','u'):'プ',
     ('p','e'):'ペ',
     ('p','o'):'ポ',
     ('v','u'):'ヴ',
     ('f','u'):'フ',
     ('y','a'):'ヤ',
     ('y','u'):'ユ',
     ('y','e'):'イェ',
     ('y','o'):'ヨ',
     ('w','a'):'ワ',
     ('ky','a'):'キャ',
     ('ky','u'):'キュ',
     ('ky','e'):'キェ',
     ('ky','o'):'キョ',
     ('ry','a'):'リャ',
     ('ry','u'):'リュ',
     ('ry','e'):'リェ',
     ('ry','o'):'リョ',
     ('sh','a'):'シャ',
     ('sh','u'):'シュ',
     ('sh','e'):'シェ',
     ('sh','o'):'ショ',
    ('ch','a'):'チャ',
    ('ch','u'):'チュ',
    ('ch','e'):'チェ',
    ('ch','o'):'チョ',
    ('ny','a'):'ニャ',
    ('ny','u'):'ニュ',
    ('ny','e'):'ニェ',
    ('ny','o'):'ニョ',
    ('my','a'):'ミャ',
    ('my','u'):'ミュ',
    ('my','e'):'ミェ',
    ('my','o'):'ミョ',
    ('gy','a'):'ギャ',
    ('gy','u'):'ギュ',
    ('gy','e'):'ギェ',
    ('gy','o'):'ギョ',
    ('hy','a'):'ヒャ',
    ('hy','u'):'ヒュ',
    ('hy','e'):'ヒェ',
    ('hy','o'):'ヒョ',
    ('py','a'):'ピャ',
    ('py','u'):'ピュ',
    ('py','e'):'ピェ',
    ('py','o'):'ピョ',
    ('by','a'):'ビャ',
    ('by','u'):'ビュ',
    ('by','e'):'ビェ',
    ('by','o'):'ビョ',
    ('dy','a'):'デャ',
    ('dy','u'):'デュ',
    ('dy','o'):'デョ',
    ('jy','a'):'テャ',
    ('jy','u'):'テュ',
    ('jy','o'):'テョ',
    ('j','a'):'ジャ',
    ('j','u'):'ジュ',
    ('j','e'):'ジェ',
    ('j','o'):'ジョ',
    ('f','a'):'ファ',
    ('f','i'):'フィ',
    ('f','e'):'フェ',
    ('f','o'):'フォ',
    ('w','i'):'ウィ',
    ('w','e'):'ウェ',
    ('w','o'):'ウォ',
    ('d','i'):'ディ',
    ('d','u'):'ドゥ',
    ('v','a'):'ヴァ',
    ('v','i'):'ヴィ',
    ('v','e'):'ヴェ',
    ('v','o'):'ヴォ',
    ('kw','a'):'クァ',
    ('kw','i'):'クィ',
    ('kw','e'):'クェ',
    ('kw','o'):'クォ',
    ('sw','a'):'スァ',
    ('sw','e'):'スェ',
    ('sw','o'):'スォ',
    ('bw','a'):'ブァ',
    ('bw','i'):'ブィ',
    ('bw','e'):'ブェ',
    ('bw','o'):'ブォ',
    ('pw','a'):'プァ',
    ('pw','i'):'プィ',
    ('pw','e'):'プェ',
    ('pw','o'):'プォ',
    ('ts','a'):'ツァ',
    ('ts','i'):'ツィ',
    ('ts','e'):'ツェ',
    ('ts','o'):'ツォ',
    ('nw','a'):'ヌァ',
    ('nw','i'):'ヌィ',
    ('nw','e'):'ヌェ',
    ('nw','o'):'ヌォ',
    ('mw','a'):'ムァ',
    ('mw','i'):'ムィ',
    ('mw','e'):'ムェ',
    ('mw','o'):'ムォ',
    ('rw','a'):'ルァ',
    ('rw','i'):'ルィ',
    ('rw','e'):'ルェ',
    ('rw','o'):'ルォ',
    ('gw','a'):'グァ',
    ('gw','i'):'グィ',
    ('gw','e'):'グェ',
    ('gw','o'):'グォ',
    ('zw','a'):'ズァ',
    ('zw','i'):'ズィ',
    ('zw','e'):'ズェ',
    ('zw','o'):'ズォ',
    ('vy','a'):'ヴャ',
    ('vy','u'):'ヴュ',
    ('vy','o'):'ヴョ',
    ('fy','a'):'フャ',
    ('fy','u'):'フュ',
    ('fy','o'):'フョ',
    ('q',''):'ッ',
     }

def rom2kana_c(cc, vv):
    if len(cc) > 0:
        if cc[-1] == ':':
            cc = cc[:-1]
        if len(vv) > 0 and vv[-1] == ':':
            vv = vv[:-1]
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
