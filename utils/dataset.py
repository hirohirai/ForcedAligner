#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2022/04/15

    Pythonで学ぶ音声合成　からの移植です
"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from omegaconf import OmegaConf

from .var import DataIdList

import logging

from utils.world_cof import WorldCof, StatsCof

# ログの設定
logger = logging.getLogger(__name__)


def pad_1d(x, max_len, constant_values=0):
    """Pad a 1d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        (0, max_len - len(x)),
        mode="constant",
        constant_values=constant_values,
    )
    return x


def pad_2d(x, max_len, constant_values=0):
    """Pad a 2d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


def pad_3d(x, max_len, constant_values=0):
    """Pad a 3d-tensor.

    Args:
        x (torch.Tensor): tensor to pad
        max_len (int): maximum length of the tensor
        constant_values (int, optional): value to pad with. Default: 0

    Returns:
        torch.Tensor: padded tensor
    """
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


class MgcSpcDataset_expand(Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output fileso
    """

    def __init__(self, in_paths_mgc, in_paths_spc, out_paths, stats_mgc, stats_spc):
        self.in_paths_mgc = in_paths_mgc
        self.in_paths_spc = in_paths_spc
        self.out_paths = out_paths
        self.mae = [1, 2, 4]
        self.usiro = [1, 2, 4]

        self.wcof = WorldCof(stats_mgc)
        self.scof = StatsCof(stats_spc)

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        ind = np.load(self.in_paths_mgc[idx])
        mgc = self.wcof.encode(ind['in_feat'])
        ind = np.load(self.in_paths_spc[idx])
        spc = self.scof.encode(ind['in_feat'])
        mgcs = []
        spcs = []
        for ix in self.mae:
            m_ = np.pad(mgc[:-ix],[[ix,0],[0,0]],'edge')
            s_ = np.pad(spc[:-ix],[[ix,0],[0,0]],'edge')
            mgcs.insert(0, m_)
            spcs.insert(0, s_)
        mgcs.append(mgc)
        spcs.append(spc)
        for ix in self.usiro:
            m_ = np.pad(mgc[ix:],[[0,ix],[0,0]],'edge')
            s_ = np.pad(spc[ix:],[[0,ix],[0,0]],'edge')
            mgcs.append(m_)
            spcs.append(s_)

        mgc = np.array(mgcs, dtype=np.float32).transpose(1, 0, 2)
        spc = np.array(spcs, dtype=np.float32).transpose(1, 0, 2)

        odat = np.load(self.out_paths[idx])
        return mgc, spc, odat['target'], idx

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.out_paths)


class MgcSpcDataset_frame(Dataset):  # type: ignore
    """Dataset for numpy files
    ノーマライズと次元の拡張はpreprocess1.pyで実行
    fid_list: F101,A01:3;7;9 or F101,A01:3:7
    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output fileso
    """

    def __init__(self, in_dir, out_dir, fid_list):
        self.in_paths = []
        self.out_paths = []
        self.frames = []
        for ee in fid_list:
            sp = ee[0]
            eee = ee[1].split(':')
            fn = eee[0]
            self.in_paths.append(f'{in_dir}/{sp}/{fn}.npz')
            self.out_paths.append(f'{out_dir}/{sp}/{fn}.npz')

            if len(eee) == 3:
                self.frames.append(range(int(eee[1]), int(eee[2])))
            else:
                nums = eee[1].split(';')
                self.frames.append([int(nn) for nn in nums])

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        ind = np.load(self.in_paths[idx])
        mgc = ind['mgc']
        spc = ind['spc']
        outd = np.load(self.out_paths[idx])
        tgt = outd['target']

        return mgc[self.frames[idx]], spc[self.frames[idx]], tgt[self.frames[idx]], idx

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.out_paths)


class MfccDataset(Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output fileso
    """

    def __init__(self, in_paths, out_paths, fid_list, in_feats):
        self.in_paths = []
        self.out_paths = []
        self.in_feats = in_feats
        for fid in fid_list:
            self.in_paths.append(f'{in_paths}/{fid[0]}/{fid[1]}.npz')
            self.out_paths.append(f'{out_paths}/{fid[0]}/{fid[1]}.npz')

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        idat = np.load(self.in_paths[idx])
        odat = np.load(self.out_paths[idx])
        return idat[self.in_feats], odat['target'], idx

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.out_paths)


class MgcSpcDataset(Dataset):  # type: ignore
    """Dataset for numpy files

    Args:
        in_paths (list): List of paths to input files
        out_paths (list): List of paths to output fileso
    """

    def __init__(self, in_paths, out_paths, fid_list, in_feats):
        self.in_paths = []
        self.out_paths = []
        self.in_feats = in_feats
        for fid in fid_list:
            self.in_paths.append(f'{in_paths}/{fid[0]}/{fid[1]}.npz')
            self.out_paths.append(f'{out_paths}/{fid[0]}/{fid[1]}.npz')

    def __getitem__(self, idx):
        """Get a pair of input and target

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        idat = np.load(self.in_paths[idx])
        odat = np.load(self.out_paths[idx])
        return idat[self.in_feats[0]], idat[self.in_feats[1]], odat['target'], idx

    def __len__(self):
        """Returns the size of the dataset

        Returns:
            int: size of the dataset
        """
        return len(self.out_paths)


def collate_fn_1para(batch):
    """Collate function

    Args:
        batch (list): List of tuples of the form (inputs, targets).

    Returns:
        tuple: Batch of inputs, targets, and lengths.
    """
    lengths = [len(x[1]) for x in batch]
    max_len = max(lengths)
    x1_batch = torch.stack([torch.from_numpy(pad_3d(x[0], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_1d(x[1], max_len)) for x in batch])
    l_batch = torch.tensor(lengths, dtype=torch.long)
    fid_batch = [x[2] for x in batch]
    return x1_batch, None, y_batch, l_batch, fid_batch


def collate_fn_2para(batch):
    """Collate function

    Args:
        batch (list): List of tuples of the form (inputs, targets).

    Returns:
        tuple: Batch of inputs, targets, and lengths.
    """
    lengths = [len(x[2]) for x in batch]
    max_len = max(lengths)
    x1_batch = torch.stack([torch.from_numpy(pad_3d(x[0], max_len)) for x in batch])
    x2_batch = torch.stack([torch.from_numpy(pad_3d(x[1], max_len)) for x in batch])
    y_batch = torch.stack([torch.from_numpy(pad_1d(x[2], max_len)) for x in batch])
    l_batch = torch.tensor(lengths, dtype=torch.long)
    fid_batch = [x[3] for x in batch]
    return x1_batch, x2_batch, y_batch, l_batch, fid_batch


def get_data_loaders0(cfg):
    """Get data loaders for training and validation.

    Args:
        cfg (dict): Data configuration.

    Returns:
        dict: Data loaders.
    """
    if cfg.train.batch_size > 1:
        logger.error(f"batch_size {cfg.train.batch_size} is not SUPPORTED !!")
        return
    data_loaders = {}

    for phase in ["train", "eval"]:
        if not cfg.data[phase]:
            continue
        logger.info(OmegaConf.to_yaml(cfg))
        fids = DataIdList(filename=cfg.data[phase].id_list)
        in_dir = Path(f'{cfg.data.base_dir}/{cfg.data.input_dir}')
        out_dir = Path(f'{cfg.data.base_dir}/{cfg.data.target_dir}')

        in_feats_paths = [in_dir / f"{fid[0]}/{fid[1]}.npz" for fid in fids]
        in_feats_paths2 = [in_dir / f"{fid[0]}/{fid[1]}_stft.npz" for fid in fids]
        out_feats_paths = [out_dir / f"{fid[0]}/{fid[1]}.npz" for fid in fids]
        wstats = f"{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[0]}"
        sstats = f"{cfg.data.base_dir}/{cfg.data.stats_dir}/{cfg.data.stats_file[1]}"

        dataset = MgcSpcDataset(in_feats_paths, in_feats_paths2, out_feats_paths, wstats, sstats)
        data_loaders[phase] = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate_fn_2para,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders

def get_data_loaders1(cfg):
    """Get data loaders for training and validation.
    preprocess1実行後
    Args:
        cfg (dict): Data configuration.

    Returns:
        dict: Data loaders.
    """
    data_loaders = {}

    for phase in ["train", "eval"]:
        if not cfg.data[phase]:
            continue
        logger.info(OmegaConf.to_yaml(cfg))
        fids = DataIdList(filename=cfg.data[phase].id_list)
        in_dir = Path(f'{cfg.data.train_dir}/{cfg.data.input_dir}')
        out_dir = Path(f'{cfg.data.train_dir}/{cfg.data.target_dir}')

        if len(cfg.data.use_stats)==1 and cfg.data.use_stats[0] in ['mgc', 'mfcc']:
            dataset = MfccDataset(in_dir, out_dir, fids, cfg.data.use_stats[0])
            collate_fn = collate_fn_1para
        else:
            #dataset = MgcSpcDataset_frame(in_dir, out_dir, fids)
            dataset = MgcSpcDataset(in_dir, out_dir, fids, cfg.data.use_stats)
            collate_fn = collate_fn_2para
        bsize = 1 if phase.startswith("eval") else cfg.train.batch_size
        data_loaders[phase] = DataLoader(
            dataset,
            batch_size=bsize,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=cfg.train.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders
