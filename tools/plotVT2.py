#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/11/02

"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.animation import ArtistAnimation
from PIL import Image

from utils.TextGrid import TextGrid
from utils.world_cof import StatsCof

# ログの設定
logger = logging.getLogger(__name__)

def read_rtmri_png(dir_, fnnum):
    gfn = glob.glob(f'{dir_}/{fnnum}/*.png')
    nn = len(gfn)
    fnb = f'{dir_}/{fnnum}/{fnnum}'
    imgs = []
    print(f'read {nn}')
    for ix in range(nn):
        fn = f'{fnb}_{ix:04}.png'
        im_ = Image.open(fn)
        imgs.append(np.asarray(im_))

    return imgs

def read_Img(dir_):
    fn = glob.glob(f'{dir_}/*.npy')
    nn = len(fn)
    imgs = []
    for ix in range(nn):
        fn = f'{dir_}/{ix:03}.npy'
        im_ = np.load(fn)
        imgs.append(im_)

    return imgs


def find_vtix(tim, vt_time, width=1/27.12*0.8):
    idx = (np.abs(vt_time - tim)).argmin()
    if np.abs(vt_time[idx] - tim) > width:
        return -1
    return idx

def plot_vtpolygon(xy, col, ax=None):
    if ax:
        return ax.plot(xy[::2], xy[1::2], color=col)
    else:
        return plt.plot(xy[::2], xy[1::2], color=col)

def calc_err(tgt, pred):
    err = np.mean(np.abs(tgt - pred))
    return err


class VTShape:
    def __init__(self, dat):
        self.x = []
        self.y = []
        vt_ed = [80, 110, 160, 220, 248, 276, 336]
        self.col = 'rrrrrrr'
        st = 0
        for ed in vt_ed:
            self.x.append(dat[st:ed:2])
            self.y.append(dat[st+1:ed:2])
            st = ed

    def plot(self, ax):
        retg = []
        for ix in range(len(self.x)):
            g = ax.plot(self.x[ix], self.y[ix], color=self.col[ix])
            retg.append(g[0])

        return retg


class FindIx:
    def __init__(self, tg1, tg2):
        if tg2 is None:
            self.diff_ix = 0
        else:
            tg1ix = int(tg1.get_frame()[0].text.split(':')[1])
            tg2ix = int(tg2.get_frame()[0].text.split(':')[1])
            self.diff_ix = tg1ix - tg2ix

    def find_ix(self, ix):
        retix = ix + self.diff_ix
        return retix


def main_ani(args):
    imgs = read_rtmri_png(args.im_dir,args.fn)
    tg = TextGrid(f'{args.tg_dir}/{args.fn}.TextGrid')
    tg_time = np.array([(frame.xmax + frame.xmin)/2 for frame in tg.get_frame()])
    vtfn = args.fn.upper()
    vtdat = np.loadtxt(f'{args.vt_dir}/{vtfn}.dat', delimiter=',')
    tg_vt = TextGrid(f'{args.vt_tg_dir}/{vtfn}.TextGrid')
    findix = FindIx(tg, tg_vt)

    anime_flg = True if len(args.anime)>0 else False

    fig, ax = plt.subplots()
    ax.axis('off')

    frames = []
    for ix in range(len(tg_time)):
        v_ix = findix.find_ix(ix)
        g1 = ax.imshow(imgs[ix], animated=anime_flg, cmap='gray')
        ph_ix = tg.find('phoneme', tg_time[ix])
        g2 = plt.text(5,250, tg.get_phoneme(ph_ix).text, color='w', size='large')

        gall = [g1, g2]
        if 0 <= v_ix < len(vtdat):
            vt = VTShape(vtdat[v_ix])
            g3 = vt.plot(ax)
            #g3 = plot_vtpolygon(vtdat[v_ix], 'r', ax)
            gall.extend(g3)


        frames.append(gall)

        if not anime_flg:
            plt.show(block=False)
            a=input()
            if (len(a)>0 and
                    a[0] == 'e'):
                break
            print(ix)
            ax.cla()

    if anime_flg:
        print(len(frames))
        ani = ArtistAnimation(fig, frames, interval=1000/args.fps, repeat=False)
        #ani.save(args.anime, writer='imagemagick')
        ani.save(args.anime)

if __name__ == "__main__":
    # Parse Arguments
    # RTMRI_DIR = '/home/oem/GitHub/rtmri-atr503'
    RTMRI_DIR = '../DBS_/rt-atr503'
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fn', default='a01')
    parser.add_argument('--im_dir', default=f'/home/hirai/Dropbox/realTimeMRI/ATR503/s1/PNG/ATR503_sentences_selected')
    parser.add_argument('--tg_dir', default=f'/home/hirai/Dropbox/realTimeMRI/ATR503/s1/TextGrid')
    parser.add_argument('--vt_dir', default=f'{RTMRI_DIR}/dat/s1/all_2')
    parser.add_argument('--vt_tg_dir', default=f'{RTMRI_DIR}/TextGrid/s1')
    parser.add_argument('--fps', type=float, default=27.1739)
    parser.add_argument('--anime', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--log', default='')
    args = parser.parse_args()

    if args.debug:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

    main_ani(args)

