#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/03/29

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import matplotlib.pyplot as plt


# ログの設定
logger = logging.getLogger(__name__)


def main(args):
    plt.plot([1,2],[3,4])
    plt.show()
    plt.plot([10, 20], [30, 40])
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
