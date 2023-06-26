#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: 
    Data: 
   一部は、ttslearnからのコピー　
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
import shutil
import random
from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

import numpy as np

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_data_loaders1 as get_data_loaders

# ログの設定
logger = logging.getLogger(__name__)


def init_seed(seed):
    """Initialize random seed.

    Args:
        seed (int): random seed

        Pythonで学ぶ音声合成　からの移植です
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_epochs_with_optional_tqdm(tqdm_mode, nepochs):
    """Get epochs with optional progress bar.

    Args:
        tqdm_mode (str): Progress bar mode.
        nepochs (int): Number of epochs.

    Returns:
        iterable: Epochs.

        Pythonで学ぶ音声合成　からの移植です
    """
    if tqdm_mode == "tqdm":
        from tqdm import tqdm

        epochs = tqdm(range(1, nepochs + 1), desc="epoch")
    else:
        epochs = range(1, nepochs + 1)

    return epochs


def make_pad_mask(lengths, maxlen=None):
    """Make mask for padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask

        Pythonで学ぶ音声合成　からの移植です
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def make_non_pad_mask(lengths, maxlen=None):
    """Make mask for non-padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length. If None, use max value of lengths.

    Returns:
        torch.ByteTensor: mask

        Pythonで学ぶ音声合成　からの移植です
    """
    return ~make_pad_mask(lengths, maxlen)


def num_trainable_params(model):
    """Count the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): Model to count the number of trainable parameters.

    Returns:
        int: Number of trainable parameters.

        Pythonで学ぶ音声合成　からの移植です
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])



def save_checkpoint(
    out_dir, model, optimizer, epoch, is_best=False, postfix=""
):
    """Save a checkpoint.

    Args:
        out_dir (str): Output directory.
        model (nn.Module): Model.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch.
        is_best (bool, optional): Whether or not the current model is the best.
            Defaults to False.
        postfix (str, optional): Postfix. Defaults to "".

        Pythonで学ぶ音声合成　からの移植です
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_best:
        path = out_dir / f"best_loss{postfix}.pth"
    else:
        path = out_dir / "epoch{:04d}{}.pth".format(epoch, postfix)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

    logger.info(f"Saved checkpoint at {path}")
    if not is_best:
        shutil.copyfile(path, out_dir / f"latest{postfix}.pth")


def set_epochs_based_on_max_steps_(train_config, steps_per_epoch):
    """Set epochs based on max steps.

    Args:
        train_config (TrainConfig): Train config.
        steps_per_epoch (int): Number of steps per epoch.
        logger (logging.Logger): Logger.

        Pythonで学ぶ音声合成　からの移植です
    """
    logger.info(f"Number of iterations per epoch: {steps_per_epoch}")

    if train_config.max_train_steps < 0:
        # Set max_train_steps based on nepochs
        max_train_steps = train_config.nepochs * steps_per_epoch
        train_config.max_train_steps = max_train_steps
        logger.info(
            "Number of max_train_steps is set based on nepochs: {}".format(
                max_train_steps
            )
        )
    else:
        # Set nepochs based on max_train_steps
        max_train_steps = train_config.max_train_steps
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        train_config.nepochs = epochs
        logger.info(
            "Number of epochs is set based on max_train_steps: {}".format(epochs)
        )

    logger.info(f"Number of epochs: {train_config.nepochs}")
    logger.info(f"Number of iterations: {train_config.max_train_steps}")

"""
def calc_loss_mse_seq(out_feats, pred_out_feats, lengths):
    mask = make_non_pad_mask(lengths).unsqueeze(-1).to(pred_out_feats.device)
    pred_out_feats_msk = pred_out_feats.masked_select(mask)
    out_feats = out_feats.masked_select(mask)

    # 損失の計算
    loss = nn.MSELoss()(pred_out_feats_msk, out_feats)

    return loss
"""
def clac_delta(feats):
    #一個前の差分を作る
    delta_feats = feats[1:]-feats[:-1]

    return delta_feats



class CalcLoss:
    def __init__(self):
        self.calc_loss = self.calc_loss_

    def calc_loss_(self, preds, targets, lengths):
        loss = nn.CrossEntropyLoss()(preds, targets)
        return loss

    def __call__(self, preds, targets, lengths):
        return self.calc_loss(preds, targets, lengths)


def setup(config, device):
    """Setup for traiining

    Args:
        config (dict): configuration for training
        device (torch.device): device to use for training

    Returns:
        (tuple): tuple containing model, optimizer, learning rate scheduler,
            data loaders, tensorboard writer, and logger.

    .. note::

        Pythonで学ぶ音声合成　からの移植です
    """
    logger.info(f"PyTorch version: {torch.__version__}")

    # CUDA 周りの設定
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark
        cudnn.deterministic = config.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")
        if torch.backends.cudnn.version() is not None:
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    logger.info(f"Random seed: {config.seed}")
    init_seed(config.seed)

    # モデルのインスタンス化
    model = hydra.utils.instantiate(config.model.netG).to(device)
    logger.info(model)
    logger.info(
        "Number of trainable params: {:.3f} million".format(
            num_trainable_params(model) / 1000000.0
        )
    )

    # (optional) 学習済みモデルの読み込み
    # ファインチューニングしたい場合
    pretrained_checkpoint = config.train.pretrained.checkpoint
    if pretrained_checkpoint is not None and len(pretrained_checkpoint) > 0:
        logger.info(
            "Fine-tuning! Loading a checkpoint: {}".format(pretrained_checkpoint)
        )
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    # Optimizer
    optimizer_class = getattr(optim, config.train.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **config.train.optim.optimizer.params
    )

    # 学習率スケジューラ
    if config.train.optim.lr_scheduler.name is not None:
        lr_scheduler_class = getattr(
            optim.lr_scheduler, config.train.optim.lr_scheduler.name
        )
        lr_scheduler = lr_scheduler_class(
            optimizer, **config.train.optim.lr_scheduler.params
        )
    else:
        lr_scheduler = None

    # DataLoader
    data_loaders = get_data_loaders(config)

    set_epochs_based_on_max_steps_(config.train, len(data_loaders["train"]))

    # Tensorboard の設定
    writer = SummaryWriter(to_absolute_path(config.train.log_dir))

    # config ファイルを保存しておく
    out_dir = Path(to_absolute_path(config.train.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.yaml", "w") as f:
        OmegaConf.save(config.model, f)
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)


    # if config.data.process == 'seq':
    #     calc_loss = calc_loss_mse_seq
    # else:
    #     calc_loss = calc_loss_mse_frame
    calc_loss = CalcLoss()

    return model, optimizer, lr_scheduler, data_loaders, writer, calc_loss
