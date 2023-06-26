#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/03/13

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F

# ログの設定
logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 3)  # pytorchの仕様のため、出力層の活性化関数は省略

    # 順伝播
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def make_data(
    N = 500, # 1クラスあたりのデータ数
    D = 2, # データの次元
    K = 3 # クラス数
    ):

    X = np.zeros((N*K,D)) # データ行列 (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # ラベル

    np.random.seed(0) #シード値固定
    #半径、角度を変化させながらデータ行列に代入
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # 半径
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # 角度
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    #3×3の単位行列を生成してラベルのワンホットエンコーディング
    n_values = np.max(y) + 1
    Y = np.eye(n_values)[y]

    rix = np.arange(N*K)
    np.random.shuffle(rix)

    return X[rix], Y[rix], y[rix]


def save_data(X,Y,y):
    batch_size=10
    y_ = np.array(y, dtype=np.int64)
    for ii in range(round(len(y_)*0.8/batch_size)):
        fn = f'data_t/in_test/s1/A{ii:03}'
        np.savez_compressed(fn, mgc=None, spc=None, mfcc=X[ii * batch_size:(ii + 1) * batch_size])
        fn = f'data_t/tgt_test/s1/A{ii:03}'
        np.savez_compressed(fn, target=y_[ii * batch_size:(ii + 1) * batch_size])

    fn = f'data_t/in_test/s1/A00'
    np.savez_compressed(fn, mgc=None, spc=None, mfcc=X[round(len(y_)*0.8):])
    fn = f'data_t/tgt_test/s1/A00'
    np.savez_compressed(fn, target=y_[round(len(y_)*0.8):])


def expand(feats, mae=[1,2,], usiro=[1,2,]):
    ofts = []
    for ix in mae:
        m_ = np.pad(feats[:-ix], [[ix, 0], [0, 0]], 'edge')
        ofts.insert(0, m_)
    ofts.append(feats)
    for ix in usiro:
        m_ = np.pad(feats[ix:], [[0, ix], [0, 0]], 'edge')
        ofts.append(m_)
    ofeats = np.array(ofts, dtype=np.float32).transpose(1, 0, 2)

    return ofeats


cols='rgb'
def main(args):
    X, Y, Y2 = make_data()
    X_ = expand(X, [1, 2, 3, 4, 6, 8, 10], [1, 2, 3, 4, 6, 8, 10])
    save_data(X_, Y, Y2)
    # データを描画
    plt.scatter(X_[:,7, 0], X_[:,7, 1], c=Y2, s=40, cmap=plt.cm.Spectral)
    plt.show()

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y2, test_size=0.2)
    train_X = torch.FloatTensor(train_X)
    train_Y = torch.LongTensor(train_Y)
    test_X = torch.FloatTensor(test_X)
    test_Y = torch.LongTensor(test_Y)

    train = TensorDataset(train_X, train_Y) #train_X、train_Yを一つにまとめる
    #訓練用データのDataloaderを作成
    train_dataloader = DataLoader(
        train,
        batch_size=300,
        shuffle=True
    )

    # インスタンス化
    net = Net()

    # 損失関数の設定(クロスエントロピー誤差)
    criterion = nn.CrossEntropyLoss()  # この中でソフトマックス関数と同じ処理をしている

    # 最適化手法の選択(SGD)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    y_axis_list = []  # 損失プロット用y軸方向リスト

    loopn = 200
    #訓練ループ
    for epoch in range(loopn):
        for batch, label in train_dataloader:
            optimizer.zero_grad()

            t_p = net(batch)

            loss = criterion(t_p,label)

            loss.backward()

            optimizer.step()

        y_axis_list.append(loss.detach().numpy())#y軸方向のリストに損失の値を代入

        if epoch % 10 == 0:#10エポック毎に損失の値を表示
            print("epoch: %d  loss: %f" % (epoch+1 ,float(loss)))

    x_axis_list = [num for num in range(loopn)]#損失プロット用x軸方向リスト

    #損失の描画
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_axis_list,y_axis_list)
    plt.show()


    with torch.no_grad():# 試験用データでは勾配を計算しない
        pred_labels = [] # 各バッチごとの結果格納用

        for x in test_X:
            pred = net(x) #モデルの出力
            #argmax関数で出力における最大値のインデックスを取得し、ワンホットエンコーディングされたラベルに変換
            if torch.argmax(pred) == torch.tensor(0) :
                pred_labels.append([1.,0.,0.])

            elif torch.argmax(pred) == torch.tensor(1):
                pred_labels.append([0.,1.,0.])

            else:
                pred_labels.append([0.,0.,1.])

    pred_labels = np.array(pred_labels) #numpy arrayに変換

    # データを描画
    plt.scatter(test_X[:, 0], test_X[:, 1], c=pred_labels, s=40, cmap=plt.cm.Spectral)
    plt.show()

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('file')
    # parser.add_argument('-s', '--opt_str', default='')
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

    main(args)
