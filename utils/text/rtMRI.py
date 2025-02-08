#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""

'''
roms = ('sp', 'a', 'i', 'u', 'e', 'o', 'N', 'j',  # 8
        'k',  'g',  's',  'z',  't',  'c', 'd',  'n',  'h',  'f',  'b',  'p',  'm',  'r',  # 14
        'kj', 'gj', 'sj', 'zj', 'tj', 'cj','dj', 'nj', 'hj', 'fj', 'bj', 'pj', 'mj', 'rj',  # 14
        'w', 's_', 'z_', 't_', 'c_', 'Q',  # 6
        '<cl>', '<j>') # 2 44

devoi = {'':0, 'A':1, 'I':2, 'U':3, 'E':4, 'O':5}
lvow = {'aH':1, 'iH':2, 'uH':3, 'eH':4, 'oH':5}
spX = {'sp0':0, 'sp1':0, 'sp2':0, 'sp3':0}
'''

NUM_A = 0
#NUM_A=5

# auto seg用
roms = ('sp', 'a', 'i', 'u', 'e', 'o') # 6 0
if NUM_A == 5:
    roms += ('A', 'I', 'U', 'E', 'O'), # 5 6
roms += ('N', 'j',  # 2 6
        'k',  'g',  's',  'z',  't', 'd',  'n',  'h',  'b',  'p',  'm',  'r',  # 12 8
        'kj', 'gj', 'sj', 'zj', 'tj', 'dj', 'nj', 'hj', 'bj', 'pj', 'mj', 'rj',  # 12 20
        'w', 'Q',  # 2 32
        '<cl>', '<j>') # 2 34 36


devoi = {'A':NUM_A+1, 'I':NUM_A+2, 'U':NUM_A+3, 'E':NUM_A+4, 'O':NUM_A+5}
lvow = {'aH':1, 'iH':2, 'uH':3, 'eH':4, 'oH':5}
lvow2 = {'AH':NUM_A+1, 'IH':NUM_A+2, 'UH':NUM_A+3, 'EH':NUM_A+4, 'OH':NUM_A+5}
spX = {'':0, 'sp0':0, 'sp1':0, 'sp2':0, 'sp3':0}

alp = {'c':NUM_A+12, 'f':NUM_A+15, 's_':NUM_A+10, 'z_':NUM_A+11, 't_':NUM_A+12, 'c_':NUM_A+12, 'cj':NUM_A+24, 'fj':NUM_A+27}

rom_to_id = {ss:ix for ix, ss in enumerate(roms)}
rom_to_id.update(devoi)
rom_to_id.update(lvow)
rom_to_id.update(lvow2)
rom_to_id.update(spX)
rom_to_id.update(alp)





CSJ_PTK = ('k', 't', 'c', 'p',
           'kj', 'tj', 'cj', 'pj',
           't_', 'c_',)

CSJ_BDG = ('g', 'd', 'b',
           'gj', 'dj', 'bj',)

CL_ph = ['k', 'g', 't', 'c', 'd', 'b', 'p', 'kj', 'gj', 'cj', 't_', 'c_', 'tj', 'dj', 'bj', 'pj', 'z', 'z_', 'zj']
J_ph = ['kj', 'sj', 'cj', 'tj', 'hj', 'fj', 'pj', 'gj', 'zj', 'dj', 'nj', 'bj', 'mj', 'rj']

# len(roms) 42