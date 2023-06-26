#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""
import os


def mk_spk_dirs(root, id_list):
    spks = id_list.get_spkid_dirs()
    for spk in spks:
        os.makedirs(f'{root}/{spk}', exist_ok=True)
