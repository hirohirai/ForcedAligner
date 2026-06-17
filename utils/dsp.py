#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/04/25

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging

import numpy as np
from scipy import interpolate

# ログの設定
logger = logging.getLogger(__name__)


def interpF0(f0, rate=0.005):
    f0_ = f0[f0>0]
    if f0[0] < 10:
        f0[0] = f0_[0]
    if f0[-1] < 10:
        f0[-1] = f0_[-1]
    xp = np.arange(0, len(f0)) * rate
    xpd = xp[f0>0]
    ypd = f0[f0>0]
    intp_func = interpolate.interp1d(xpd, ypd, kind="quadratic", fill_value='extrapolate')
    return intp_func(xp)


def median1d(arr, k):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype=int) - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1
    return np.median(arr[idx], axis=0)