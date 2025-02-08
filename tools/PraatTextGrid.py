#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    hoge_tools.py
"""
import sys, os
import logging
import argparse
import subprocess
import PySimpleGUI as sg


# ログの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
logger.addHandler(stream_handler)


# const
# NUM_XXX = 4

#sendpraat_cmd = "C:\\Users\\hiroh\\Downloads\\sendpraat-win.exe"
sendpraat_cmd = "sendpraat"

class TextGrid:
    def __init__(self, wvdir, tgdir, mergin, wvext='.wav', wv_UL_flg=''):
        self.wvdir = wvdir
        self.wvext = wvext
        self.tgdir = tgdir
        self.ph_mergin = mergin
        self.wv_UL_flg = wv_UL_flg

    def open_window(self, base):
        logger.debug(str(base))
        self.basename = base[0]
        if self.wv_UL_flg in ['U', 'L']:
            self.bn_wav = self.basename.upper() if self.wv_UL_flg=='U' else self.basename.lower()
        else:
            self.bn_wav = self.basename
        self.wvfn = self.wvdir + '/' + self.bn_wav + self.wvext
        self.tgfn = self.tgdir + '/' + self.basename + '.TextGrid'
        cmd = "{} praat \"Read from file... {}\" \"Read from file... {}\"".format(sendpraat_cmd, self.wvfn, self.tgfn)
        subprocess.run(cmd, shell=True)
        cmd = "{} praat \"selectObject: \\\"Sound {}\\\", \\\"TextGrid {}\\\"\"".format(sendpraat_cmd, self.bn_wav, self.basename)
        subprocess.run(cmd, shell=True)
        cmd = f"{sendpraat_cmd} praat \"View & Edit\""
        subprocess.run(cmd, shell=True)
        if len(base) > 2 and base[1]>0:
            st = base[1] - self.ph_mergin
            if st<0:
                st=0.0
            ed = base[2] + self.ph_mergin
            cmd = f'{sendpraat_cmd} praat'
            cmd += f' "editor: \\\"TextGrid {self.basename}\\\""'
            cmd += f' "Select: {st},{ed}"'
            cmd += f' "Zoom to selection"'
            cmd += f' "endeditor"'
            logger.debug(f'sendpraat: {cmd}')
            subprocess.run(cmd, shell=True)

    def save_textGrid(self):
        cmd = "{} praat \"selectObject: \\\"TextGrid {}\\\"\"".format(sendpraat_cmd, self.basename)
        subprocess.run(cmd, shell=True)
        cmd = "{} praat \"Save as text file... {}\"".format(sendpraat_cmd, self.tgfn)
        subprocess.run(cmd, shell=True)

    def remove_window(self):
        cmd = "{} praat \"selectObject: \\\"Sound {}\\\", \\\"TextGrid {}\\\"\"".format(sendpraat_cmd, self.bn_wav, self.basename)
        subprocess.run(cmd, shell=True)
        cmd = f"{sendpraat_cmd} praat \"Remove\""
        subprocess.run(cmd, shell=True)


def main_open_one(args):
    textgrid = TextGrid(args.wvdir, args.tgdir, args.mergin, args.wvext)
    textgrid.open_window(args.kanalist)
    


def main(args):
    """ 

    """
    logger.debug("main start")
    f_list = []
    for ll in open(args.kanalist):
        ee = ll.strip().split()
        f_list.append(ee[0])

    textgrid = TextGrid(args.wvdir, args.tgdir, args.mergin, args.wvext, args.wvUL)

    if args.start.isdecimal():
        stix = int(args.start)
    else:
        stix = -1
    cix = stix
    while 1:
        print("{}: {}".format(cix, f_list[cix]))
        textgrid.open_window(f_list[cix])
        print("save:s ｓ / save&next:n ｎ / next: > / back: < / quit:q")
        cmd = sys.stdin.readline().strip()
        if cmd == 's' or cmd == 'ｓ' or cmd == '2':
            textgrid.save_textGrid()
        elif cmd == 'n' or cmd == 'ｎ' or cmd == '0':
            textgrid.save_textGrid()
            textgrid.remove_window()
            cix += 1
        elif cmd == '>' or cmd == '3':
            textgrid.remove_window()
            cix += 1
        elif cmd == '<' or cmd == '1':
            textgrid.remove_window()
            if cix >0:
                cix -= 1
        elif cmd == 'q':
            break

class TgFiles:
    def __init__(self, fn):
        self.body = {}
        self.lbls = []
        with open(fn) as ifs:
            tmpfn = {}
            for ll in ifs:
                ee = ll.strip().split()
                if ee[0] in tmpfn:
                    tmpfn[ee[0]] += 1
                    lbl = f'{ee[0]}:{tmpfn[ee[0]]}'
                else:
                    tmpfn[ee[0]] = 0
                    lbl = ee[0]
                self.lbls.append(lbl)
                if len(ee) == 3:
                    self.body[lbl] = [ee[0], float(ee[1]), float(ee[2])]
                else:
                    self.body[lbl] = [ee[0], -1.0, -1.0]

    def get_posi(self, lbl):
        for ix, ee in enumerate(self.lbls):
            if ee == lbl:
                return ix

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.body[self.lbls[item]]
        else:
            return self.body[item]



class ScPosition:
    def __init__(self, len_, sz):
        self.len = len_
        self.sz = sz

    def get_position(self, posi):
        posi = float(posi - self.sz /2.0)
        if posi < 0:
            return 0.0
        elif posi > self.len - self.sz:
            posi = self.len - self.sz
        return posi / self.len

def main_gui(args):
    """

    """
    logger.debug("main start")
    initlayout = [[[sg.Text('WavDir:'), sg.InputText(default_text=args.wvdir, key='wavdir'),
                    sg.FolderBrowse(key='wavdirFB')]],
                  [[sg.Text('TextGridDir:'), sg.InputText(default_text=args.tgdir, key='tgdir'),
                    sg.FolderBrowse(key='tgdirFB  ')]],
                  [[sg.Text('file list:'), sg.InputText(default_text=args.kanalist, key='kanalist'),
                    sg.FileBrowse(key='kanalistFB')],
                   [[sg.Text('start index:'), sg.InputText(default_text=args.start, key='stposi')]],
                  [sg.Submit()]],
                  ]

    initwindow = sg.Window('init', initlayout)
    event, values = initwindow.read()
    logger.debug(initwindow['wavdir'].get())
    logger.debug(initwindow['tgdir'].get())
    logger.debug(initwindow['kanalist'].get())
    kanalist = initwindow['kanalist'].get()
    wvdir = initwindow['wavdir'].get()
    tgdir = initwindow['tgdir'].get()
    stposi = initwindow['stposi'].get()
    initwindow.close()



    if stposi.isdecimal():
        stix = int(stposi)
    else:
        stix = -1

    tg_list = TgFiles(kanalist)
    f_list = tg_list.lbls

    textgrid = TextGrid(wvdir, tgdir, args.mergin, args.wvext, args.wvUL)
    col1 = [[sg.Button(' up ')], [sg.Button('Save')], [sg.Button('Next')], [sg.Button('down')], [sg.Button('Read')]]
    layout = [[sg.Listbox(values=f_list, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, size=(10,30), key='-FileList-', enable_events=True), sg.Column(col1)]]
    window = sg.Window('TextGrid Edit', layout)
    stvalue = f_list[stix]
    lbox = window['-FileList-']
    lbox.set_value(stvalue)
    #print('st:', stvalue, val)
    textgrid.open_window(tg_list[stix])

    scp = ScPosition(len(f_list), 30)
    lbox.set_vscroll_position(scp.get_position(stix))

    while True:  # Event Loop
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        elif event == 'Save':
            stix = lbox.get_indexes()[-1]
            logger.debug(f'save {stix}')
            textgrid.save_textGrid()
        elif event == 'Next':
            stix = lbox.get_indexes()[-1]
            logger.debug(f'next {stix}')
            textgrid.save_textGrid()
            if stix < len(f_list) - 1:
                stix = stix + 1
                lbox.set_value([f_list[stix],])
                textgrid.remove_window()
                textgrid.open_window(tg_list[stix])
        elif event == ' up ':
            stix = lbox.get_indexes()[-1]
            logger.debug(f'up {stix}')
            if 0 < stix:
                stix = stix - 1
                lbox.set_value([f_list[stix],])
                textgrid.remove_window()
                textgrid.open_window(tg_list[stix])
        elif event == 'down':
            stix = lbox.get_indexes()[-1]
            logger.debug(f'down {stix}')
            if stix < len(f_list) - 1:
                stix = stix + 1
                logger.debug(f'down: {f_list[stix]}')
                lbox.set_value([f_list[stix],])
                textgrid.remove_window()
                textgrid.open_window(tg_list[stix])
        elif event == 'Read':
            stix = lbox.get_indexes()[-1]
            logger.debug(f'read {stix}')
            textgrid.remove_window()
            textgrid.open_window(tg_list[stix])

        #lbox.set_vscroll_position(scp.get_position(stix))

    window.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-s", "--start", default='0')
    parser.add_argument("-m", "--mergin", type=float, default=0.1)
    parser.add_argument("--wvUL", default='')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nogui", action="store_true")
    parser.add_argument("--wvdir", default='/home/hirai/work_local/Speech/DBS_/rtmri-atr503/wav/s1')
    parser.add_argument("--tgdir", default='/home/hirai/work_local/Speech/DBS_/rtmri-atr503/TextGrid/s1')
    parser.add_argument("--kanalist", default='./atr503.lst')
    parser.add_argument("--wvext", default='.wav')
    args = parser.parse_args()

    if args.verbose:
        stream_handler.setLevel(logging.INFO)
    elif args.debug:
        stream_handler.setLevel(logging.DEBUG)

    if os.path.exists(args.kanalist):
        if args.nogui:
            main(args)
        else:
            main_gui(args)
    else:
        if len(args.kanalist)==0 or (len(args.kanalist)>4 and args.kanalist[-4:] == '.lst'):
            main_gui(args)
        else:
            main_open_one(args)
