#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""
cnvRom = {'wo':'o', 'he':'e'}

roms = ('sp', 'a', 'i', 'u', 'e', 'o', 'N',
      'k', 's', 'sh', 't', 'ch', 'ts', 'n', 'h', 'm', 'r',
      'g', 'z', 'j', 'd', 'b', 'p', 'v', 'f', 'y', 'w',
      'ky', 'ry', 'ny', 'my', 'gy', 'hy', 'py', 'by', 'dy', 'jy',
      'kw', 'sw', 'bw', 'pw', 'nw', 'mw', 'rw', 'gw', 'zw', 'vy', 'fy',
      'q')

devoi = {'':0, 'A':1, 'I':2, 'U':3, 'E':4, 'O':5}
lvow = {'a:':1, 'i:':2, 'u:':3, 'e:':4, 'o:':5}

rom_to_id = {ss:ix for ix, ss in enumerate(roms)}
rom_to_id.update(devoi)
rom_to_id.update(lvow)

ROM_PTK = ('k', 't', 'ch', 'ts', 'p',
       'ky', 'py',
      'kw', 'pw')

ROM_BDG = ('g', 'z', 'j', 'd', 'b', 'v',
      'gy', 'by', 'dy', 'jy',
      'bw', 'gw', 'zw',)

# 47  len(roms)
