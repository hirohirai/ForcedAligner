#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TextGrid.py
    Author: hirai
    Data: 2019/08/29
"""

import sys
import os
import os.path
import argparse
import logging
import re
import codecs

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text import kana


# ログの設定
logger = logging.getLogger(__name__)

itemnum_ptn = re.compile(r".*\[\s*(\d+)\s*].*")
MIN_DUR = 0.01

class Interval:
    def __init__(self, kana='', st=0.0, ed=0.0):
        self.xmin = st
        self.xmax = ed
        self.text = kana

    def set(self, bufs, ix):
        while ix < len(bufs):
            ll = bufs[ix].strip()
            if len(ll) == 0:
                ix += 1
                continue
            ee = ll.split()
            if ee[0].startswith("intervals") or ee[0].startswith("item"):
                return ix
            ix += 1
            if ee[0] == "xmin":
                self.xmin = float(ee[-1])
            elif ee[0] == "xmax":
                self.xmax = float(ee[-1])
            elif ee[0] == "text":
                txt = '　'.join(ee[2:])
                self.text = txt.strip('"')

        return ix

    def __str__(self):
        if self.xmin == 0.0:
            obuf = "            xmin = 0 \n"
        else:
            obuf = "            xmin = {} \n".format(self.xmin)
        obuf += "            xmax = {} \n".format(self.xmax)
        obuf += '            text = "{}" \n'.format(self.text)
        return obuf


class Item:
    def __init__(self, ct='', nm=''):
        self.class_type = ct
        self.name = nm
        self.xmin = 0.0
        self.xmax = 0.0
        self.intervals = []

    def __str__(self):
        obuf = "        class = {} \n".format(self.class_type)
        obuf += "        name = {} \n".format(self.name)
        if self.xmin == 0.0:
            obuf += "        xmin = 0 \n"
        else:
            obuf += "        xmin = {} \n".format(self.xmin)
        obuf += "        xmax = {} \n".format(self.xmax)
        if len(self.intervals)>0:
            obuf += "        intervals: size = {} \n".format(len(self.intervals))
            for ix, it in enumerate(self.intervals):
                obuf += "        intervals [{}]:\n".format(ix+1)
                obuf += str(it)
        return obuf

    def set(self, bufs, ix):
        while ix < len(bufs):
            ll = bufs[ix].strip()
            ee = ll.split()
            if ll.startswith("item"):
                return ix
            ix += 1
            if ll.startswith("class"):
                self.class_type = ee[-1]
            elif ll.startswith("name"):
                self.name = ee[-1]
            elif ll.startswith("xmin"):
                self.xmin = float(ee[-1])
            elif ll.startswith("xmax"):
                self.xmax = float(ee[-1])
            elif ll.startswith("intervals"):
                m = itemnum_ptn.match(ll)
                if m:
                    nn = int(m.group(1))
                    if len(self.intervals) == nn-1:
                        intv = Interval()
                        ix = intv.set(bufs, ix)
                        self.intervals.append(intv)
        return ix

    def clear_intervals(self):
        self.intervals = []

    def append_interval(self, kana , st, ed):
        self.intervals.append(Interval(kana, st, ed))
        if self.xmin > st:
            self.xmin = st
        if self.xmax < ed:
            self.xmax = ed

    def insert_interval(self, ix, kana, st, ed):
        self.intervals.insert(ix, Interval(kana, st, ed))

    def find_st(self, tim):
        for ix, it in enumerate(self.intervals):
            if it.xmin >= tim:
                break
        if ix >0:
            if tim - self.intervals[ix-1].xmin < self.intervals[ix].xmin - tim:
                ix -= 1
        return ix

    def find_ed(self, tim):
        for ix, it in enumerate(self.intervals):
            if it.xmax >= tim:
                break
        if ix > 0:
            if tim - self.intervals[ix - 1].xmax < self.intervals[ix].xmax - tim:
                ix -= 1
        return ix

    def find(self, tim):
        for ix, it in enumerate(self.intervals):
            if it.xmax >= tim:
                break
        if self.intervals[ix].xmin <= tim:
            return ix
        else:
            return None


class TextGrid:
    def __init__(self, ifs=None):
        if ifs is None:
            self.item = []
            self.FileType = '"ooTextFile"'
            self.ObjectClass = '"TextGrid"'
            self.xmin = 0
            self.xmax = 0
            return
        if isinstance(ifs, str):
            if ifs == '-':
                ifs = sys.stdin
            else:
                ifs = codecs.open(ifs, 'r', 'utf-8')
        bufs = ifs.readlines()
        self.item = []
        ix = 0
        while ix < len(bufs):
            ll = bufs[ix].strip()
            if len(ll) == 0:
                ix += 1
                continue
            elif ll.startswith("item"):
                m = itemnum_ptn.match(ll)
                ix += 1
                if m:
                    nn = int(m.group(1))
                    if len(self.item) == nn-1:
                        itm = Item()
                        ix = itm.set(bufs, ix)
                        self.item.append(itm)
                    else:
                        logger.error("item format error")
            else:
                ee = ll.split()
                if ll.startswith("File type"):
                    self.FileType = ee[-1]
                elif ll.startswith("Object class"):
                    self.ObjectClass = ee[-1]
                elif ee[0] == "xmin":
                    self.xmin = float(ee[-1])
                elif ee[0] == "xmax":
                    self.xmax = float(ee[-1])
                ix += 1

    def get_item_ix(self, name):
        for ix, itm in enumerate(self.item):
            if name in itm.name:
                return ix
        return -1

    def clear_phoneme(self):
        self.item = [itm for itm in self.item if itm.name != '"phoneme"']

    def clear_words(self):
        self.item = [itm for itm in self.item if itm.name != '"kana"']

    def clear_swords(self):
        self.item = [itm for itm in self.item if itm.name != '"word"']

    def clear_sents(self):
        self.item = [itm for itm in self.item if itm.name != '"trans"']

    def insert_word(self, ix, kana, st, ed):
        for itm in self.item:
            if itm.name == '"kana"':
                itm.insert_interval(ix, kana, st, ed)
                break

    def insert_sword(self, ix, wrd, st, ed):
        for itm in self.item:
            if itm.name == '"word"':
                itm.insert_interval(ix, wrd, st, ed)
                break

    def insert_phoneme(self, ix, phn, st, ed):
        for itm in self.item:
            if itm.name == '"phoneme"':
                itm.insert_interval(ix, phn, st, ed)
                break

    def append_phoneme(self, phn, st, ed):
        if len(self.item) == 0 or self.item[0].name != '"phoneme"':
            self.item.insert(0, Item('"IntervalTier"', '"phoneme"'))
        self.item[0].append_interval(phn, st, ed)

    def append_word(self, kana, st, ed):
        ix = -1
        if len(self.item) == 0:
            self.item.append(Item('"IntervalTier"', '"kana"'))
            ix = 0
        elif len(self.item) == 1:
            if self.item[0].name == '"phoneme"':
                self.item.append(Item('"IntervalTier"', '"kana"'))
                ix = 1
            elif self.item[0].name == '"trans"':
                self.item.insert(0, Item('"IntervalTier"', '"kana"'))
                ix = 0
            elif self.item[0].name == '"kana"':
                ix = 0
            else:
                ix = 1
        elif len(self.item) == 2:
            if self.item[0].name == '"kana"':
                ix = 0
            elif self.item[1].name == '"kana"':
                ix = 1
            else:
                self.item.insert(1, Item('"IntervalTier"', '"kana"'))
                ix = 1
        else:
            if self.item[1].name == '"kana"':
                ix = 1
            elif self.item[0].name == '"kana"':
                ix = 0
            elif self.item[2].name == '"kana"':
                ix = 2

        logger.debug(f'appword {ix} lenitem {len(self.item)}')
        self.item[ix].append_interval(kana, st, ed)
        if self.xmin > st:
            self.xmin = st
        if self.xmax < ed:
            self.xmax = ed


    def append_sword(self, sword, st, ed):
        ix = -1
        for ii,it in enumerate(self.item):
            if it.name == '"word"':
                ix = ii
                break
            elif it.name == '"trans"':
                self.item.insert(ii, Item('"IntervalTier"', '"word"'))
                ix = ii
                break
        if ix < 0:
            self.item.append(Item('"IntervalTier"', '"word"'))

        logger.debug(f'appword {ix} lenitem {len(self.item)}')
        self.item[ix].append_interval(sword, st, ed)
        if self.xmin > st:
            self.xmin = st
        if self.xmax < ed:
            self.xmax = ed

    def append_sent(self, text, st=-1, ed=-1):
        if len(self.item) == 0:
            self.item.append(Item('"IntervalTier"', '"trans"'))
        elif self.item[-1].name != '"trans"':
            self.item.append(Item('"IntervalTier"', '"trans"'))
        if ed < 0:
            self.item[-1].append_interval(text, self.xmin, self.xmax)
        else:
            self.item[-1].append_interval(text, st, ed)

    def addFrameNum(self, stposi, fps=27.1739, fn='', frate=-1, LastFull=True):
        if fps == 0.0:
            return
        import math
        if frate<0:
            frate = 1/ fps
        if len(self.item) > 0 and self.item[-1].name == '"frame"':
            self.item = self.item[:-1]
        self.item.append(Item('"IntervalTier"', '"frame"'))
        self.xmin = 0.0
        stn = math.floor((self.xmin + stposi)/ frate + 0.00001) + 1
        #num = math.floor(stposi / frate)
        if LastFull:
            edn = math.floor((self.xmax + stposi)/ frate + 0.00001)
        else:
            edn = math.ceil((self.xmax + stposi)/ frate - 0.00001)
        st = 0.0
        #ed = (num +1) * frate - stposi
        ed = stn * frate - stposi
        for nn in range(stn, edn+1):
            if ed > self.xmax:
                ed = self.xmax
            lbl = f'{fn}:{nn}' if len(fn)>0 else f'{nn}'
            self.item[-1].append_interval(lbl, st, ed)
            st = ed
            ed += frate

    def copyFrameNum(self, tg):
        if len(self.item) > 0 and self.item[-1].name == '"frame"':
            self.item = self.item[:-1]
        self.item.append(Item('"IntervalTier"', '"frame"'))
        for frm in tg.get_frame():
            self.item[-1].append_interval(frm.text, frm.xmin, frm.xmax)


    def addStEd(self, ixs=[0,1,2,3,4]):
        if isinstance(ixs, int):
            ixs = [ixs,]
        for ix in ixs:
            if ix < len(self.item):
                if self.item[ix].name == '"frame"':
                    continue
                self.item[ix].xmin = self.xmin
                self.item[ix].xmax = self.xmax
                if len(self.item) > 0 and len(self.item[ix].intervals) > 0:
                    if self.item[ix].intervals[0].text != "":
                        if ix==0 and self.item[ix].intervals[0].text == "<cl>":
                            self.item[ix].intervals[0].text = "#,<cl>"
                            self.item[ix].intervals[0].xmin = 0
                        else:
                            if self.item[ix].intervals[0].xmin == 0.0:
                                if self.item[ix].intervals[0].xmax > MIN_DUR:
                                    self.item[ix].intervals[0].xmin = MIN_DUR
                            if self.item[ix].intervals[0].xmin > 0:
                                if self.item[ix].intervals[0].text == '#' or self.item[ix].intervals[0].text == '#,<cl>':
                                    self.item[ix].intervals[0].xmin = 0
                                else:
                                    intv = Interval()
                                    intv.xmax = self.item[ix].intervals[0].xmin
                                    intv.text = '#'
                                    self.item[ix].intervals.insert(0, intv)
                    else:
                        self.item[ix].intervals[0].text = "#"
                        self.item[ix].intervals[0].xmin = 0

                    if self.item[ix].intervals[-1].text != "":
                        if self.item[ix].intervals[-1].xmax < self.xmax:
                            if self.item[ix].intervals[-1].text == "#":
                                self.item[ix].intervals[-1].xmax = self.xmax
                            else:
                                intv = Interval()
                                intv.xmin = self.item[ix].intervals[-1].xmax
                                intv.xmax = self.xmax
                                intv.text = "#"
                                self.item[ix].intervals.append(intv)
                    else:
                        self.item[ix].intervals[-1].text = "#"
                        self.item[ix].intervals[-1].xmax = self.xmax

    def correct_times(self, ixs=[0,1,2,3,4]):
        for ix in ixs:
            if ix < len(self.item):
                last = 0.0
                for ii in self.item[ix].intervals:
                    if ii.xmin != last:
                        ii.xmin = last
                    last = ii.xmax

    def correct_word(self):
        for it in self.get_word():
            it.text = it.text.replace('ヅ', 'ズ')

    def set_xmax_xmin(self, ed, st=0):
        self.xmin = st
        self.xmax = ed
        for itm in self.item:
            itm.xmin = st
            itm.xmax = ed

    def add_time(self, addTime, ixs=[0,1,2,3,4]):
        self.xmin += addTime
        self.xmax += addTime
        if self.xmin < 0.0:
            self.xmin = 0.0
        if isinstance(ixs, int):
            ixs = [ixs,]
        else:
            ixs_ = []
            for ii in ixs:
                if ii < len(self.item):
                    ixs_.append(ii)
            ixs = ixs_
        for ix in ixs:
            self.item[ix].xmin += addTime
            self.item[ix].xmax += addTime
            if self.item[ix].xmin < 0.0:
                self.item[ix].xmin = 0.0
            for ii in self.item[ix].intervals:
                ii.xmin += addTime
                ii.xmax += addTime
                if ii.xmin < 0.0:
                    ii.xmin = 0.0

    def find_st(self, name, tim):
        for it_ in self.item:
            if it_.name == f'"{name}"':
                return it_.find_st(tim)
        return None

    def find_ed(self, name, tim):
        for it_ in self.item:
            if it_.name == f'"{name}"':
                return it_.find_ed(tim)
        return None

    def find(self, name, tim):
        for it_ in self.item:
            if it_.name == f'"{name}"':
                return it_.find(tim)
        return None

    def find_ph(self, cph, lph=None, nph=None):
        if self.item[0].name != '"phoneme"':
            return None
        rets =[]
        if not isinstance(cph, list):
            cph = [cph,]
        if lph and not isinstance(lph, list):
            lph = [lph,]
        if nph and not isinstance(nph, list):
            nph = [nph,]
        st = 1 if lph else 0
        ed = len(self.item[0].intervals)-1 if nph else len(self.item[0].intervals)
        for ii in range(st, ed):
            if self.item[0].intervals[ii].text in cph:
                if lph and self.item[0].intervals[ii - 1].text not in lph:
                    continue
                if nph and self.item[0].intervals[ii + 1].text not in nph:
                    continue
                rets.append(ii)
        return rets

    def join_ph(self, ix):
        if self.item[0].name != '"phoneme"' or len(self.item[0].intervals) <= ix-1:
            return
        self.item[0].intervals[ix].xmax = self.item[0].intervals[ix + 1].xmax
        self.item[0].intervals[ix].text = self.item[0].intervals[ix].text + ',' + self.item[0].intervals[ix+1].text
        del self.item[0].intervals[ix+1]

    def __str__(self):
        obuf = "File type = {}\n".format(self.FileType)
        obuf += "Object class = {}\n\n".format(self.ObjectClass)
        if self.xmin == 0.0:
            obuf += "xmin = 0 \n".format(self.xmin)
        else:
            obuf += "xmin = {} \n".format(self.xmin)
        obuf += "xmax = {} \n".format(self.xmax)
        if len(self.item)>0:
            obuf += "tiers? <exists> \nsize = {} \nitem []: \n".format(len(self.item))
            for ix, itm in enumerate(self.item):
                obuf += "    item [{}]:\n".format(ix+1)
                obuf += str(itm)
        return obuf

    def getStEd(self, inx=0):
        if isinstance(inx, str):
            inx = self.get_item_ix(inx)
        st = -1
        ed = -1
        if len(self.item) > inx:
            for ii in self.item[inx].intervals:
                if ii.text != "" and ii.text != "sp" and ii.text != '#':
                    st = ii.xmin
                    break
            for ix in range(len(self.item[inx].intervals)):
                if self.item[inx].intervals[-1-ix].text != "" and self.item[inx].intervals[-1 - ix].text != "sp" and self.item[inx].intervals[-1 - ix].text != "#":
                    ed = self.item[inx].intervals[-1-ix].xmax
                    break
        return st, ed

    def get_kanaSent(self):
        outk = kana.KanaSent()
        for it_ in self.item:
            if it_.name == '"kana"':
                for ii in it_.intervals:
                    if len(ii.text) > 0:
                        outk.add_word_textGrid(ii.text)
                outk.set_bound()
                break
        for it_ in self.item:
            if it_.name == '"word"':
                swrds = []
                for ii in it_.intervals:
                    if len(ii.text) > 0 and ii.text not in ['sp', '#', 'sp0', 'sp1', 'sp2']:
                        swrds.append(ii.text)
                outk.add_sword(swrds)
                break

        return outk

    def get_jeitaKana(self):
        outk = self.get_kanaSent()
        return outk.get_jeitaKana()

    def get_kana(self):
        outk = self.get_kanaSent()
        return outk.get_juliusKana()

    def get_sampa(self):
        outk = self.get_kanaSent()
        return outk.get_sampa()

    def get_input_feature(self):
        return self.get_sampa()

    def get_text(self, sep=''):
        out_text = []
        for itm in self.item:
            if itm.name == '"trans"':
                for ii in itm.intervals:
                    if len(ii.text) > 0 and ii.text != '#':
                        out_text.append(ii.text)
                return sep.join(out_text)
        return None

    def get_size(self):
        return len(self.get_word())

    def get_word(self, ii=None):
        for itm in self.item:
            if itm.name == '"kana"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None

    def get_sent(self, ii=None):
        for itm in self.item:
            if itm.name == '"trans"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None

    def get_trans(self, ii=None):
        for itm in self.item:
            if itm.name == '"trans"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None


    def get_sword(self, ii=None):
        for itm in self.item:
            if itm.name == '"word"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None

    def get_phoneme(self, ii=None):
        for itm in self.item:
            if itm.name == '"phoneme"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None

    def get_frame(self, ii=None):
        for itm in self.item:
            if itm.name == '"frame"':
                if ii is None:
                    return itm.intervals
                else:
                    return itm.intervals[ii]
        return None


def main(args):
    tg = TextGrid(args.file1)
    out = args.sep.join(tg.get_input_feature())
    print(out)

def main2(args):
    tg = TextGrid(args.file1)
    tg.addStEd()
    tg.correct_times()
    print(tg)

def main3(args):
    tg = TextGrid(args.file1)
    print(tg.get_text())


def main4(args):
    tg = TextGrid(args.file1)
    out = args.sep.join(tg.get_jeitaKana())
    print(out)

def main5(args):
    tg = TextGrid(args.file1)
    #print(tg.get_text())
    for ix in range(tg.get_size()):
        if tg.get_word(ix).text == '':
            continue
        print(tg.get_word(ix).text, end=' ')
        print(tg.get_word(ix).xmin, end=' ')
        print(tg.get_word(ix).xmax)

def main6(args):
    tg = TextGrid(args.file1)
    print(tg.getStEd())

def main7(args):
    tg = TextGrid(args.file1)
    tg.add_time(args.addTime)
    tg.correct_times()
    print(tg)


def main8(args):
    outs = []
    tg = TextGrid(args.file1)
    for ph in tg.get_phoneme():
        if len(ph.text)>0:
            outs.append(ph.text)
    print(args.sep.join(outs))

def main9(args):
    tg = TextGrid(args.file1)
    out = args.sep.join(tg.get_kana())
    print(out)

def main10(args):
    import Tts
    ifs = codecs.open(args.sword, 'r', 'utf-8')
    l = ifs.readline()
    swords = l.strip().split()
    tg = TextGrid(args.file1)
    tg.correct_word()
    ks = tg.get_kanaSent()
    ks.add_sword(swords)
    tts = Tts.Tts()
    tts.from_kanaSent(ks)
    tts.set_time(tg, True)
    tts.xmin = tg.xmin
    tts.xmax = tg.xmax
    tts.text = tg.get_text()
    tgo = Tts.tts_to_textGrid(tts)
    for ix in range(4):
        tgo.addStEd(ix)

    print(str(tgo))


def main11(args):
    import Tts
    tg = TextGrid(args.file1)
    tts = Tts.textGrid_to_Tts(tg)
    print(tts)


def main12(args):
    import Tts
    tg = TextGrid(args.file1)
    tts = Tts.textGrid_to_Tts(tg, force_clJ=True)
    tg2 = Tts.tts_to_textGrid(tts)
    tg2.correct_times()
    print(str(tg2))


def main13(args):
    tg = TextGrid(args.file1)
    tg.xmin = 0.0
    tg.xmax = args.ed - args.st
    for ix in range(4):
        tg.item[ix].xmin = tg.xmin
        tg.item[ix].xmax = tg.xmax
        for ii , it in enumerate(tg.item[ix].intervals):
            if it.xmax > args.st:
                break
        del tg.item[ix].intervals[:ii]
        for ii, it in enumerate(tg.item[ix].intervals):
            if it.xmin > args.ed:
                break
        del tg.item[ix].intervals[ii:]
        for ii, it in enumerate(tg.item[ix].intervals):
            it.xmin -= args.st
            if it.xmin < tg.xmin:
                it.xmin = tg.xmin
            it.xmax -= args.st
            if it.xmax > tg.xmax:
                it.xmax = tg.xmax
    if len(tg.item[0].intervals[0].text)>1 and tg.item[0].intervals[0].text[-1]=='#':
        tg.item[0].intervals[0].text= '#'
    if len(tg.item[0].intervals[-1].text)>1 and tg.item[0].intervals[0].text[0]=='#':
        tg.item[0].intervals[-1].text= '#'
    print(tg)

def main14(args):
    tg = TextGrid(args.file1)
    tg.addFrameNum(args.st, args.fps)
    print(tg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("-s", "--sep", default="")
    parser.add_argument("--addTime", type=float, default=0.0)
    parser.add_argument("-m", "--mode", default="input_feature")
    parser.add_argument("--sword", default="")
    parser.add_argument("--st", type=float, default=0.0)
    parser.add_argument("--ed", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=27.1739)
    args = parser.parse_args()

    if args.mode == "input_feature":
        main(args)
    elif args.mode == "addStEd":
        main2(args)
    elif args.mode == "text":
        main3(args)
    elif args.mode == "kana":
        main4(args)
    elif args.mode == "juliusKana":
        main9(args)
    elif args.mode == "sted":
        main6(args)
    elif args.mode == "kanaTime":
        main5(args)
    elif args.mode == "addTime":
        main7(args)
    elif args.mode == "phoneme":
        main8(args)
    elif args.mode == "addSword":
        main10(args)
    elif args.mode == "tts":
        main11(args)
    elif args.mode == "reformat":
        main12(args)
    elif args.mode == "split":
        main13(args)
    elif args.mode == "addFrame":
        main14(args)

    else:
        logger.warning("Mode error")
        main3(args)
