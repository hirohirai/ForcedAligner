#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/09/12

"""
import logging

# ログの設定
logger = logging.getLogger(__name__)

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PhonemeProbSC(nn.Module):
    def __init__(self, stft_order=15, stft_cnn_dims=[64,], ksize=[5,], pksize=[3,], plst=[2,], mgc_slice=15, mgc_order=28, mgc_dims=[256,], fc_indim=3968, fc_dims=[128,], out_dim=44):
        super().__init__()

        # stft 畳込み
        self.stft_lyr = nn.Sequential()
        self.fl = nn.Flatten(1, -1)


        last_dim = stft_order
        for ix, odim in enumerate(stft_cnn_dims):
            self.stft_lyr.add_module(f'stft_cn{ix}', nn.Conv1d(last_dim, odim, ksize[ix]))
            self.stft_lyr.add_module(f'stft_bn{ix}', nn.BatchNorm1d(odim))
            self.stft_lyr.add_module(f'stft_ac{ix}', nn.ReLU())
            self.stft_lyr.add_module(f'stft_pl{ix}', nn.MaxPool1d(pksize[ix], stride=plst[ix]))
            last_dim = odim

        self.stft_lyr.add_module('stft_dout', nn.Dropout(p=0.25))

        self.mgc_lyr = nn.Sequential()
        last_dim = mgc_slice * mgc_order
        for ix, odim in enumerate(mgc_dims):
            self.mgc_lyr.add_module(f'mgc_fc{ix}', nn.Linear(last_dim, odim))
            self.mgc_lyr.add_module(f'mgc_ac{ix}', nn.ReLU())
            self.mgc_lyr.add_module(f'mgc_dout{ix}', nn.Dropout(p=0.5))
            last_dim = odim

        self.mgc_lyr.add_module(f'mgc_fco', nn.Linear(last_dim, fc_indim))
        self.mgc_lyr.add_module(f'mgc_aco', nn.ReLU())
        self.mgc_lyr.add_module(f'mgc_douto', nn.Dropout(p=0.5))

        self.fc_lyr = nn.Sequential()
        last_dim = fc_indim
        for ix, odim in enumerate(fc_dims):
            self.fc_lyr.add_module(f'fc{ix}', nn.Linear(last_dim, odim))
            self.fc_lyr.add_module(f'ac{ix}', nn.ReLU())
            self.fc_lyr.add_module(f'dout{ix}', nn.Dropout(p=0.5))
            last_dim = odim
        self.fc_lyr.add_module(f'fcout', nn.Linear(last_dim, out_dim))
        # self.fc_lyr.add_module(f'softmax', nn.Softmax(dim=1))

    def forward(self, in1=None, in2=None, length_=None):
        in2 = self.stft_lyr(in2)
        #print("SPC0", spc.shape)
        in2 = self.fl(in2)
        #print("SPC1", spc.shape)

        in1 = self.fl(in1)
        in1 = self.mgc_lyr(in1)

        y = self.fc_lyr(in2 + in1)
        # y = self.fc_lyr(in1)

        return y


class PhonemeProbC(nn.Module):
    def __init__(self, h_dims=[1024,1024,1024,1024,1024,1024], mfcc_dim=13, mfcc_slice=1, out_dim=6):
        super().__init__()
        self.mfcc_lyr = nn.Sequential()
        lastdim = mfcc_dim * mfcc_slice
        for ix, hd in enumerate(h_dims):
            self.mfcc_lyr.add_module(f'mfcc_fc_in{ix}', nn.Linear(lastdim, hd))
            self.mfcc_lyr.add_module(f'mfcc_fc_ac{ix}', nn.ReLU())
            self.mfcc_lyr.add_module(f'mfcc_fc_dout{ix}', nn.Dropout(p=0.5))
            lastdim = hd
        self.mfcc_lyr.add_module(f'mfcc_fc_out', nn.Linear(lastdim, out_dim))
        #self.smax = nn.Softmax(dim=1)
        self.fl = nn.Flatten(1, -1)

    def forward(self, in1=None, in2=None, length_=None):
        #in1 = in1[:,7,:]
        x = self.fl(in1)
        x = self.mfcc_lyr(x)
        #y = self.smax(x)
        return x
