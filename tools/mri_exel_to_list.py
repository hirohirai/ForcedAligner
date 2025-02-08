#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2023/08/07

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging

# ログの設定
logger = logging.getLogger(__name__)


class SentPos:
    def __init__(self, name, fname, st, ed=None):
        self.name = name
        self.fname = fname
        self.st = st
        self.ed = ed

    def __str__(self):
        return f'{self.name} {self.st} {self.ed} {self.fname}'

    def print(self):
        print(f'{self.name} {self.fname}')

class RtmriPos:
    def __init__(self, fn=None, MyFlg=False, skip_col=1):
        self.fnums = []
        self.body = []
        if fn:
            with open(fn) as ifs:
                if MyFlg:
                    self.body.append([])
                    for ll in ifs:
                        ee = ll.strip().split()
                        if len(ee) == 6:
                            self.body[0].append(SentPos(ee[0].lower(), ee[5], ee[3], ee[4]))
                            if ee[5] not in self.fnums:
                                self.fnums.append(ee[5])
                        else:
                            self.body[0].append(SentPos(ee[0].lower(), ee[1], ee[2], ee[3]))
                            if ee[1] not in self.fnums:
                                self.fnums.append(ee[1])

                else:
                    fnum = ifs.readline().strip().split(',')
                    self.fnums = fnum[skip_col+1:]
                    for ix in range(len(self.fnums)):
                        self.body.append([])
                    for ll in ifs:
                        ee = ll.strip().split(',')
                        time = ee[skip_col]
                        for ix, sname in enumerate(ee[skip_col+1:]):
                            sname = sname.strip()
                            if len(sname)>0:
                                if len(self.body[ix])>0 and self.body[ix][-1].ed is None:
                                    self.body[ix][-1].ed = time
                                if sname != 'end':
                                    self.body[ix].append(SentPos(sname, self.fnums[ix], time))

                    for ix in range(len(self.body)):
                        if self.body[ix][-1].ed == time:
                            if '_' in self.body[ix][-1].name:
                                self.body[ix] = self.body[ix][:-1]
                            else:
                                logging.error(f"Last file of {self.body[ix][-1].name} {self.fnums[ix]}")





def main(args):
    if args.my:
        rp = RtmriPos(args.file, MyFlg=True)
    else:
        rp = RtmriPos(args.file, skip_col=args.skip)
    for opos in rp.body:
        for pp in opos:
            if args.all:
                print(pp)
            else:
                pp.print()




if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-s', '--skip', type=int, default=1)
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
    parser.add_argument('--my', action='store_true')
    parser.add_argument('--all', action='store_true')
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
