#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data:

"""
import sys
import os.path
import argparse
import logging
import pickle
import random


def main(args):
    if args.input != '-':
        ifs = open(args.input)
        ifname = os.path.splitext(args.input)
    else:
        ifs = sys.input
        ifname = ('stdin', 'txt')
    buf=[]
    for ll in ifs:
        buf.append(ll.strip())
    random.shuffle(buf)
    eix = int(len(buf) * args.rate + 0.5)
    eix2 = round((len(buf)-eix) * args.rate2) + eix
    if args.rate2 > 0 and len(buf) - eix2 <= 0 and len(buf) - eix > 2:
        eix2 = len(buf) - 1
    buf1 = buf[:eix]
    buf1.sort()
    buf2 = buf[eix:eix2]
    buf2.sort()
    buf3 = buf[eix2:]

    with open("{}_train{}".format(ifname[0], ifname[1]), 'w') as ofs:
        for ll in buf1:
            print(ll, file=ofs)

    with open("{}_eval{}".format(ifname[0], ifname[1]), 'w') as ofs:
        for ll in buf2:
            print(ll, file=ofs)

    if len(buf3) > 0:
        with open("{}_test{}".format(ifname[0], ifname[1]), 'w') as ofs:
            for ll in buf3:
                print(ll, file=ofs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="-")
    parser.add_argument("-r", "--rate", type=float, default=0.95)
    parser.add_argument("--rate2", type=float, default=0.9)
    parser.add_argument("-v", "--verbose", action='store_true')
    args = parser.parse_args()

    main(args)



