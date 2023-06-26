#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/05/22

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import glob

import pydicom

# ログの設定
logger = logging.getLogger(__name__)


def search(args):
    for fn in glob.glob(args.file):
        #print(fn)
        dicom = pydicom.read_file(fn)
        if dicom.InstanceNumber < 3:
            print(dicom.InstanceNumber, fn)


def sort(args):
    tim = {}
    for fn in glob.glob(args.file):
        dicom = pydicom.read_file(fn)
        tim[dicom.InstanceNumber] = float(dicom.AcquisitionTime)

    tim[0] = 0.0
    for ix in range(1,514):
        if ix in tim.keys():
            print(ix, tim[ix], tim[ix]- tim[ix-1])


def dump(args):
    dicom = pydicom.read_file(args.file)
    print(dicom)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-m', '--mode', default='search')
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

    if args.mode == 'search':
        search(args)
    elif args.mode == 'dump':
        dump(args)
    else:
        sort(args)
