#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:


"""
import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
import numpy as np

from utils.train_func import save_checkpoint
from utils.train_func import setup, get_epochs_with_optional_tqdm

from torch.nn import functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, 3)  # pytorchの仕様のため、出力層の活性化関数は省略

    # 順伝播
    def forward(self, x, x2, ll):
        x = x[:,7,:]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x



def save_checkpoint_data(out_dir, target, pred, epoch, fid_np, train, sfid=None, in1=None, in2=None):
    target_out_np = target.to('cpu').detach().numpy().copy()
    pred_out_np = pred.to('cpu').detach().numpy().copy()
    in1_np = in1.to('cpu').detach().numpy().copy() if in1 is not None else None
    in2_np = in2.to('cpu').detach().numpy().copy() if in2 is not None else None
    fnb = 'T' if train else 'E'

    if sfid is None:
        fn = f'{out_dir}/target_{epoch}_{fnb}'
        np.save(fn, target_out_np)
        fn = f'{out_dir}/pred_{epoch}_{fnb}'
        np.save(fn, pred_out_np)
        np.savetxt(f'{out_dir}/fid_{epoch}_{fnb}.lst', fid_np, fmt='%d')
    else:
        for ix, fid_ in enumerate(fid_np):
            if fid_ in sfid:
                fn = f'{out_dir}/target_{epoch}_{fnb}_{fid_}_{ix}'
                np.save(fn, target_out_np)
                fn = f'{out_dir}/pred_{epoch}_{fnb}_{fid_}_{ix}'
                np.save(fn, pred_out_np)
                if in1_np is not None:
                    fn = f'{out_dir}/in1_{epoch}_{fnb}_{fid_}_{ix}'
                    np.save(fn, in1_np)
                if in2_np is not None:
                    fn = f'{out_dir}/in2_{epoch}_{fnb}_{fid_}_{ix}'
                    np.save(fn, in2_np)


def debug_save(in_feats=None, in_feats2=None, out_feats=None, lengths=None, pred_out_feats=None, odir='./', exit_flg=True):
    if in_feats:
        np.save(in_feats.to('cpu').detach().numpy().copy(), f'{odir}in1.npy')
    if in_feats2:
        np.save(in_feats2.to('cpu').detach().numpy().copy(), f'{odir}in2.npy')
    if out_feats:
        np.save(out_feats.to('cpu').detach().numpy().copy(), f'{odir}target.npy')
    if lengths:
        np.save(lengths.to('cpu').detach().numpy().copy(), f'{odir}lengths.npy')
    if pred_out_feats:
        np.save(pred_out_feats.to('cpu').detach().numpy().copy(), f'{odir}pred.npy')
    if exit_flg:
        sys.exit(1)

def train_step(model, optimizer, calc_loss, train, in1, in2, targets, lengths):

    optimizer.zero_grad()

    # 順伝播
    preds = model(in1, in2, lengths)

    # debug_save(in_feats, out_feats, lengths, pred_out_feats)

    loss = calc_loss(preds, targets, lengths)

    # 逆伝播、モデルパラメータの更新
    if train:
        loss.backward()
        optimizer.step()

    return loss, preds


def train_loop(
    config, device, model, optimizer, lr_scheduler, data_loaders, writer, calc_loss
):
    #model = Net().to(device)

    out_dir = Path(config.train.out_dir)
    best_loss = torch.finfo(torch.float32).max

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in get_epochs_with_optional_tqdm(config.tqdm, config.train.nepochs):
        for phase in data_loaders.keys():
            train = phase.startswith("train")
            model.train() if train else model.eval()
            running_loss = 0
            try:
                for in1, in2, targets, lengths, fid in data_loaders[phase]:
                    if in1 is not None:
                        in1 = torch.flatten(in1, 0, 1)
                        in1 = in1.to(device)
                    if in2 is not None:
                        in2 = torch.flatten(in2, 0, 1)
                        in2 = in2.to(device)
                    targets = torch.flatten(targets, 0, 1)
                    targets = targets.to(device)

                    loss, pred_out = train_step(model, optimizer, calc_loss, train, in1, in2, targets, lengths)
                    if torch.isnan(loss):
                        logging.warning(f'loss NAN: {fid}')
                    else:
                        running_loss += loss.item()

                    if epoch % config.train.checkpoint_epoch_interval == 0:
                        fid_np = np.array(fid)
                        if train:
                            cfid = np.intersect1d(fid_np, np.array(config.train.save_fid_T))
                        else:
                            cfid = np.intersect1d(fid_np, np.array(config.train.save_fid_E))
                        if len(cfid) > 0:
                            save_checkpoint_data(out_dir, targets, pred_out, epoch, fid_np, train, cfid, in1, in2)

            except Exception as e:
                logging.error(f'{fid} {e}')


            ave_loss = running_loss / len(data_loaders[phase])
            #print(phase, running_loss, ave_loss)
            writer.add_scalar(f"Loss/{phase}", ave_loss, epoch)

            if not train and ave_loss < best_loss:
                best_loss = ave_loss
                save_checkpoint(out_dir, model, optimizer, epoch, is_best=True)

        if lr_scheduler:
            lr_scheduler.step()
        if epoch % config.train.checkpoint_epoch_interval == 0:
            save_checkpoint(out_dir, model, optimizer, epoch, is_best=False)

    save_checkpoint(out_dir, model, optimizer, config.train.nepochs)


@hydra.main(version_base='1.2',config_path="conf", config_name="config")
def my_app(config: DictConfig) -> None:
    if config.debug:
        device =torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, optimizer, lr_scheduler, data_loaders, writer, calc_loss = setup(
        config, device
    )
    train_loop(
        config, device, model, optimizer, lr_scheduler, data_loaders, writer, calc_loss
    )


if __name__ == "__main__":
    my_app()
