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
    'ヲ': ('','wo'),
    'ヱ': ('','he'),
    }

def kana2rom_v(kn):
    if kn in kana2rom_v_tbl:
        return kana2rom_v_tbl[kn]
    return None

kana2rom_c1_tbl = {
    'カ':('k','a'),
    'キ':('k','i'),
    'ク':('k','u'),
    'ケ':('k','e'),
    'コ':('k','o'),
    'サ':('s','a'),
    'シ':('sh','i'),
    'ス':('s','u'),
    'セ':('s','e'),
    'ソ':('s','o'),
    'タ':('t','a'),
    'チ':('ch','i'),
    'ツ':('ts','u'),
    'テ':('t','e'),
    'ト':('t','o'),
    'ナ':('n','a'),
    'ニ':('n','i'),
    'ヌ':('n','u'),
    'ネ':('n','e'),
    'ノ':('n','o'),
    'ハ':('h','a'),
    'ヒ':('h','i'),
    #'フ':('h','u'),
    'ヘ':('h','e'),
    'ホ':('h','o'),
    'マ':('m','a'),
    'ミ':('m','i'),
    'ム':('m','u'),
    'メ':('m','e'),
    'モ':('m','o'),
    'ラ':('r','a'),
    'リ':('r','i'),
    'ル':('r','u'),
    'レ':('r','e'),
    'ロ':('r','o'),
    'ガ':('g','a'),
    'ギ':('g','i'),
    'グ':('g','u'),
    'ゲ':('g','e'),
    'ゴ':('g','o'),
    'ザ':('z','a'),
    'ジ':('j','i'),
    'ズ':('z','u'),
    'ゼ':('z','e'),
    'ゾ':('z','o'),
    'ダ':('d','a'),
    'ヅ':('z','u'),
    'デ':('d','e'),
    'ド':('d','o'),
    'バ':('b','a'),
    'ビ':('b','i'),
    'ブ':('b','u'),
    'ベ':('b','e'),
    'ボ':('b','o'),
    'パ':('p','a'),
    'ピ':('p','i'),
    'プ':('p','u'),
    'ペ':('p','e'),
    'ポ':('p','o'),
    'ヴ':('v','u'),
    'フ':('f','u'),
    'ヤ':('y','a'),
    'ユ':('y','u'),
    'ヨ':('y','o'),
    'ワ':('w','a'),
    }

kana2rom_c2_tbl = {
    'ティ':('t','i'),
    'スィ':('s','i'),
    'トゥ':('t','u'),
    'ズィ':('z','i'),
    'イェ':('y','e'),
    'キャ':('ky','a'),
    'キュ':('ky','u'),
    'キェ':('ky','e'),
    'キョ':('ky','o'),
    'リャ':('ry','a'),
    'リュ':('ry','u'),
    'リェ':('ry','e'),
    'リョ':('ry','o'),
    'シャ':('sh','a'),
    'シュ':('sh','u'),
    'シェ':('sh','e'),
    'ショ':('sh','o'),
    'チャ':('ch','a'),
    'チュ':('ch','u'),
    'チェ':('ch','e'),
    'チョ':('ch','o'),
    'ニャ':('ny','a'),
    'ニュ':('ny','u'),
    'ニェ':('ny','e'),
    'ニョ':('ny','o'),
    'ミャ':('my','a'),
    'ミュ':('my','u'),
    'ミェ':('my','e'),
    'ミョ':('my','o'),
    'ギャ':('gy','a'),
    'ギュ':('gy','u'),
    'ギェ':('gy','e'),
    'ギョ':('gy','o'),
    'ヒャ':('hy','a'),
    'ヒュ':('hy','u'),
    'ヒェ':('hy','e'),
    'ヒョ':('hy','o'),
    'ピャ':('py','a'),
    'ピュ':('py','u'),
    'ピェ':('py','e'),
    'ピョ':('py','o'),
    'ビャ':('by','a'),
    'ビュ':('by','u'),
    'ビェ':('by','e'),
    'ビョ':('by','o'),
    'デャ':('dy','a'),
    'デュ':('dy','u'),
    'デョ':('dy','o'),
    'テャ':('jy','a'),
    'テュ':('jy','u'),
    'テョ':('jy','o'),
    'ジャ':('j','a'),
    'ジュ':('j','u'),
    'ジェ':('j','e'),
    'ジョ':('j','o'),
    'ファ':('f','a'),
    'フィ':('f','i'),
    'フェ':('f','e'),
    'フォ':('f','o'),
    'ウィ':('w','i'),
    'ウェ':('w','e'),
    'ウォ':('w','o'),
    'ディ':('d','i'),
    'ドゥ':('d','u'),
    'ヴァ':('v','a'),
    'ヴィ':('v','i'),
    'ヴェ':('v','e'),
    'ヴォ':('v','o'),
    'クァ':('kw','a'),
    'クィ':('kw','i'),
    'クェ':('kw','e'),
    'クォ':('kw','o'),
    'スァ':('sw','a'),
    'スェ':('sw','e'),
    'スォ':('sw','o'),
    'ブァ':('bw','a'),
    'ブィ':('bw','i'),
    'ブェ':('bw','e'),
    'ブォ':('bw','o'),
    'プァ':('pw','a'),
    'プィ':('pw','i'),
    'プェ':('pw','e'),
    'プォ':('pw','o'),
    'ツァ':('ts','a'),
    'ツィ':('ts','i'),
    'ツェ':('ts','e'),
    'ツォ':('ts','o'),
    'ヌァ':('nw','a'),
    'ヌィ':('nw','i'),
    'ヌェ':('nw','e'),
    'ヌォ':('nw','o'),
    'ムァ':('mw','a'),
    'ムィ':('mw','i'),
    'ムェ':('mw','e'),
    'ムォ':('mw','o'),
    'ルァ':('rw','a'),
    'ルィ':('rw','i'),
    'ルェ':('rw','e'),
    'ルォ':('rw','o'),
    'グァ':('gw','a'),
    'グィ':('gw','i'),
    'グェ':('gw','e'),
    'グォ':('gw','o'),
    'ズァ':('zw','a'),
    #'ズィ':('zw','i'),
    'ズェ':('zw','e'),
    'ズォ':('zw','o'),
    'ヴャ':('vy','a'),
    'ヴュ':('vy','u'),
    'ヴョ':('vy','o'),
    'フャ':('fy','a'),
    'フュ':('fy','u'),
    'フョ':('fy','o'),
     }

def kana2rom_c(kn):
    if kn in kana2rom_c2_tbl:
        return kana2rom_c2_tbl[kn]
    if kn in kana2rom_c1_tbl:
        return kana2rom_c2_tbl[kn]
    return None

def kana2rom_vc(kn):
    if kn in kana2rom_c2_tbl:
        return kana2rom_c2_tbl[kn]
    if kn in kana2rom_c1_tbl:
        return kana2rom_c2_tbl[kn]
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
            if roms[-1][1][-1] != ':':
                roms.append(('', roms[-1][1] + ':'))
            else:
                roms.append(('', roms[-1][1]))
        elif kn[ix] == 'ッ':
            roms.append(('q', ''))

        ix += 1

    if ix < len(kn):
        if kn[ix] in kana2rom_c1_tbl:
            roms.append(kana2rom_c1_tbl[kn[ix]])
        elif kn[ix] in kana2rom_v_tbl:
            roms.append(kana2rom_v_tbl[kn[ix]])
        elif kn[ix] == 'ー':
            roms.append(('', roms[-1][1] + ':'))
    return roms

