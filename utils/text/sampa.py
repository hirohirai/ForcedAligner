#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    sampa.py
    Author: hirai
    Data: 2019/08/30
"""
_pad = ('_',)
# 韻律記号　0,2は０型のときのアクセントの強度
pros = ('<F>||', '<R>||', '_R<F>||', '^', '!', '!0', '!2', '0', '2', '(.)', '(..)', '(...)', '|', '||', '-',)
letter = ('4','4_w','4j','C','J',
'M','N','N\\','N_w','Nj',
'P\\j','S','a','b','b_w',
'bj','d','dZ','d_w','dj',
'dz','dz_w','e','f','g',
'g_w','gj','h','i','j',
'k','k_w','kj','m','m_w',
'mj','n','n_w','o','p',
'p\\','p_w','pj','s','s_w',
't','tS','t_w','tj','ts',
'v','v_w','vj','w',)
special = (':', 'q',)

symbols = _pad + pros + letter + special

sampa_to_id = {ss:ix for ix, ss in enumerate(symbols)}
