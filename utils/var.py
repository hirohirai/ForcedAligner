#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    
    Author: hirai
    Data:

"""
import pickle
import logging
import re
import glob
import os.path
import numpy as np

from .TextGrid import TextGrid

# ログの設定
logger = logging.getLogger(__name__)


# ((spkid, fileid), data_len)
class DataSetList(list):
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            with open(filename, 'rb') as f:
                tmp = pickle.load(f)
                for id_ in tmp:
                    self.append(id_)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def get_id_list(self):
        ids = DataIdList()
        for (id, ll) in self:
            ids.append(id)

        return ids


# id = (spkid, fileid)
class DataIdList(list):
    def __init__(self, root_path=None, spk_type_ptn='(.*)/', extension='.wav', filename=None):
        super().__init__()
        if filename is not None:
            self.load(filename)
        elif root_path is not None:
            self.set_Id_list(root_path, spk_type_ptn, extension)

    def set_Id_list(self, root_path, spk_type_ptn='(.*)/', extension='.wav'):
        if root_path[-1] != '/':
            root_path += '/'
        if spk_type_ptn[0] != '(':
            spk_type_ptn = f'({spk_type_ptn})/'
        ptn = re.compile(root_path + spk_type_ptn + r'([^\s]+)' + extension)
        for filename in glob.iglob(f'{root_path}**/*{extension}', recursive=True):
            res = ptn.match(filename)
            if res:
                spk_type_id = res.group(1)
                file_id = res.group(2)
                self.append((spk_type_id, file_id))

    def load(self, filename):
        if filename[-4:] == '.csv':
            with open(filename, 'r') as f:
                for ll in f:
                    ee = ll.strip().split(',')
                    id_ = (ee[0].strip(), ee[1].strip())
                    self.append(id_)
        else:
            pass

    def save(self, filename):
        if filename[-4:] == '.csv':
            with open(filename, 'w') as f:
                for id in self:
                    print(f'{id[0]},{id[1]}', file=f)
        else:
            pass

    def get_spkid_list(self):
        ulist = set()
        for ids in self:
            ulist.add(ids[0])
        return ulist

    def get_spkid_dirs(self):
        ulist = set()
        for ids in self:
            fn = f'{ids[0]}/{ids[1]}'
            dd = os.path.dirname(fn)
            ulist.add(dd)
        return ulist

    def get_spkid_dic(self):
        dics = {}
        for ids in self:
            if ids[0] in dics:
                dics[ids[0]].append(ids[1])
            else:
                dics[ids[0]] = [ids[1],]
        return dics

    def intersection(self, idl2):
        aa = DataIdList()
        #self = [id_ for id_ in self if id_ in idl2]
        for id_ in self:
            if id_ in idl2:
                aa.append(id_)
        return aa



''' 以下、未使用
# id = ((spkid, fileid), filename)
class DataIdFnameList(list):
    def __init__(self, root_path=None, spk_type_ptn='(.*)/', extension='.TextGrid'):
        super().__init__()
        if root_path is not None:
            self.set_Id_list(root_path, spk_type_ptn, extension)

    def set_Id_list(self, root_path, spk_type_ptn='(.*)/', extension='.TextGrid'):
        if root_path[-1] != '/':
            root_path += '/'
        ptn = re.compile(root_path + spk_type_ptn + r'([^/\s]+)' + extension)
        for filename in glob.iglob(f'{root_path}**/*{extension}', recursive=True):
            res = ptn.match(filename)
            if res:
                spk_type_id = res.group(1)
                file_id = res.group(2)
                self.append(((spk_type_id, file_id), filename))


# [id] -> phone list of sentence by sampa
class TextDict(dict):
    def __init__(self, filename=None):
        super().__init__()
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'rb') as f:
                tmp = pickle.load(f)
                for kk in tmp.keys():
                    self[kk] = tmp[kk]
        else:
            pass

    def save(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            pass


# [id] -> xvector
class SpkEmbDict:
    def __init__(self, filename=None):
        super().__init__()
        self.spid2ix = {}
        self.emb_vec = []
        if filename is not None:
            self.load(filename)

    def make_dict(self, spid_fn, xvec_fn):
        with open(spid_fn, 'r') as f:
            for ll in f:
                ee = ll.strip().split(',')
                self.spid2ix[ee[0]] = int(ee[1])
        self.emb_vec = np.load(xvec_fn)
        return self

    def __getitem__(self, id_):
        return self.emb_vec[self.spid2ix[id_]]

    def load(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'rb') as f:
                [self.spid2ix, self.emb_vec] = pickle.load(f)
        else:
            pass

    def save(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'wb') as f:
                pickle.dump([self.spid2ix, self.emb_vec], f)
        else:
            pass

# [id] -> index of speaker
class SpkIdDict:
    def __init__(self, filename=None):
        super().__init__()
        self.spid2ix = {}
        if filename is not None:
            self.load(filename)

    def make_dict(self, spid_fn):
        with open(spid_fn, 'r') as f:
            for ll in f:
                ee = ll.strip().split(',')
                self.spid2ix[ee[0]] = int(ee[1])

    def __getitem__(self, id_):
        return self.spid2ix[id_]

    def load(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'rb') as f:
                self.spid2ix = pickle.load(f)
        else:
            pass

    def save(self, filename):
        if filename[-4:] == '.pkl':
            with open(filename, 'wb') as f:
                pickle.dump(self.spid2ix, f)
        else:
            pass

# create textdict and timedict
def get_from_textGrid(root_path, spkr_ptn, file_ids):
    ids = DataIdFnameList(root_path, spkr_ptn, extension='.TextGrid')

    text_dict = TextDict()
    time_dict = {}
    data_ids = DataIdList()

    for id, fn in ids:
        if id not in file_ids:
            continue

        try:
            tg = TextGrid(fn)
        except:
            logger.warning("TextGrid Read Error: " + fn)
            continue
        td = tg.get_input_feature()
        sted = tg.getStEd()
        if len(td) > 0 and sted != (-1, -1):
            text_dict[id] = td
            time_dict[id] = sted
            data_ids.append(id)
        else:
            logger.warning("Text Grid kana error: " + fn)

    data_ids.sort()
    return text_dict, time_dict, data_ids

'''