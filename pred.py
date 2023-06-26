import torch
import numpy as np

import argparse
import logging

from omegaconf import OmegaConf
from hydra.utils import instantiate
import librosa

from utils.world_cof import WorldCof, StatsCof
from preprocess1 import expand



def load_data(mgcFn=None, spcFn=None, mfccFn=None, stnum=-1, ednum=-1):
    if mgcFn:
        mgc = np.load(mgcFn)
        mgc = mgc['in_feat']
        if stnum > 0:
            a = mgc[0]
            mae = np.repeat(a[np.newaxis, :], stnum, axis=0)
            mgc = np.vstack([mae, mgc])
        if ednum > 0:
            a = mgc[-1]
            usiro = np.repeat(a[np.newaxis, :], ednum, axis=0)
            mgc = np.vstack([mgc, usiro])
    else:
        mgc = None

    if spcFn:
        spc = np.load(spcFn)
        spc = spc['in_feat']
        if stnum>0:
            a = spc[0]
            mae = np.repeat(a[np.newaxis,:], stnum, axis=0)
            spc = np.vstack([mae, spc])
        if ednum>0:
            a = spc[-1]
            usiro = np.repeat(a[np.newaxis,:], ednum, axis=0)
            spc = np.vstack([spc, usiro])
    else:
        spc = None

    if mfccFn:
        mfcc = np.load(mfccFn)
        mfcc = mfcc['in_feat']
        if stnum>0:
            a = mfcc[0]
            mae = np.repeat(a[np.newaxis,:], stnum, axis=0)
            mfcc = np.vstack([mae, mfcc])
        if ednum>0:
            a = mfcc[-1]
            usiro = np.repeat(a[np.newaxis,:], ednum, axis=0)
            mfcc = np.vstack([mfcc, usiro])
    else:
        mfcc = None

    return mgc, spc, mfcc


def conv_data(wcof=None, scof=None, mcof=None, mgc=None, spc=None, mfcc=None):
    if mgc is not None:
        mgc = wcof.encode(mgc)
        mgc_s = expand(mgc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])

    else:
        mgc_s = None

    if spc is not None:
        spc = scof.encode(spc)
        spc_s = expand(spc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    else:
        spc_s = None

    if mfcc is not None:
        mfcc = mcof.encode(mfcc)
        mfcc_s = expand(mfcc, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    else:
        mfcc_s = None

    return mgc_s, spc_s, mfcc_s


def load_model(cnf_fn, weight_fn):
    cnf = OmegaConf.load(cnf_fn)
    model = instantiate(cnf['netG'])
    pth = torch.load(weight_fn)
    model.load_state_dict(pth['state_dict'])
    return model


def make_mfcc(args):
    [wv, sr] = librosa.load(args.wav, sr=16000)
    wv = wv * 2.5
    wvpe = librosa.effects.preemphasis(wv)
    mfcc = librosa.feature.mfcc(y=wvpe, sr=sr, n_mfcc=13, hop_length=80)
    mfcc = mfcc.T
    return mfcc

def make_mgc(args):
    mgcfn = f'{args.world}.mgc'
    capfn = f'{args.world}.cap'
    f0fn = f'{args.world}.f0'
    mgc = np.fromfile(mgcfn, dtype=np.float64)
    mgc = mgc.reshape(int(len(mgc) / args.mgcP), args.mgcP)
    mgc = mgc[:, :args.mgcPo]
    cap = np.fromfile(capfn, dtype=np.float64)
    cap = cap.reshape(int(len(cap) / args.capP), args.capP)
    f0 = np.fromfile(f0fn, dtype=np.float64)
    return np.hstack([f0[:, np.newaxis], mgc, cap])

def make_spc(args):
    [wv, sr] = librosa.load(args.wav, sr=10000)
    r = np.random.randn(len(wv)) * 1e-5
    # r[wv!=0.0] = 0
    stft = np.log(np.abs(librosa.stft(wv+r, n_fft=256, hop_length=50))).T
    return stft

def make_in_feat(args):
    wcof, mcof, scof = load_cof(args)
    mgc=spc=mfcc=None
    if args.world:
        mgc = make_mgc(args)
        if args.use_spc or args.wav:
            spc = make_spc(args)
    elif args.wav:
        mfcc = make_mfcc(args)
        if args.use_spc:
            spc = make_spc(args)
    else:
        mgc, spc, mfcc = load_data(args.mgc, args.spc, args.mfcc, args.stnum, args.ednum)

    mgc, spc, mfcc = conv_data(wcof, scof, mcof, mgc, spc, mfcc)

    if mgc is not None:
        return mgc, spc
    elif mfcc is not None:
        return mfcc, spc
    else:
        return None, spc


def save_results(pred, fn):
    np.save(fn, pred.to('cpu').detach().numpy().copy())


def load_cof(args):
    wcof = WorldCof(args.wcof) if args.wcof else None
    mcof = StatsCof(args.mcof) if args.mcof else None
    scof = StatsCof(args.scof) if args.scof else None
    return wcof, mcof, scof


def main(args):
    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Model
    model = load_model(args.model, args.weights).to(device)
    model.eval()

    with torch.no_grad():

        in_feat1, in_feat2 = make_in_feat(args)

        in_feat1 = torch.from_numpy(in_feat1).to(device) if in_feat1 is not None else None
        in_feat2 = torch.from_numpy(in_feat2).to(device) if in_feat2 is not None else None

        #np.save('debug1', mfcc)
        pred = model(in_feat1, in_feat2)

        save_results(pred, args.output)

    print('Done.')


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='conf/model/dnn_mgc.yaml', type=str,
                        help='[string/path] Load model')
    parser.add_argument('--weights', '-w', default='exp/best_loss.pth', type=str,
                        help='[string/path] Load in different model Weights')
    parser.add_argument('--mcof', default='data/stats/mfcc.npz', type=str, help='load Path')
    parser.add_argument('--wcof', default='data/stats/world.npz', type=str, help='load Path')
    parser.add_argument('--scof', default='data/stats/stft.npz', type=str, help='load Path')
    parser.add_argument('--world', '-i', default='', type=str, help='load Path')
    parser.add_argument('--mgcP', default=45, type=int, help='load Path')
    parser.add_argument('--capP', default=2, type=int, help='load Path')
    parser.add_argument('--mgcPo', default=25, type=int, help='load Path')
    parser.add_argument('--wav', default='', type=str, help='load Path')
    parser.add_argument('--use_spc', '-s', action='store_true', help='load Path')
    parser.add_argument('--mgc', default='', type=str, help='load Path')
    parser.add_argument('--mfcc', default='', type=str, help='load Path')
    parser.add_argument('--spc', default='', type=str, help='load Path')
    parser.add_argument('--stnum', default=-1, type=int, help='load Path')
    parser.add_argument('--ednum', default=-1, type=int, help='load Path')
    parser.add_argument('--output', '-o', default='./pred.npy', type=str, help='Save Path')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(args)
