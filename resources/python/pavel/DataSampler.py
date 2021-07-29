"""
Data sampler

create_splits
    Reads directory with images, where each subdirectory contains images for one label. Name of the directory is the label.
    Then it creates splits for trn/val/tst data sets.
    Example of expected image structure:
    - label1/
        - file1.bmp
        - file2.bmp
    - label2/
        - file1.bmp
        - file2.bmp
    - label3/
        - file1.bmp
        - file2.bmp
        - file3.bmp

load/save
    Load or save data split into CSV file

get_label_filename
    Generator (label, path)

get_label_image
    Generator (label, image)

Pavel Krizek @ Wickon, July 2021
"""

import os
import math
import fileio as utl
import random
from collections import defaultdict
from PIL import Image

class DataSampler:
    """Data sampler
    :param path: path to data
    :param labels: map with data label name conversion to number
    :param filename: (optional) load file with splits
    :param verbose: (optional) display info
    """

    def __init__(self, path, labels, filename=None, verbose=False):
        """Init data sampler"""
        self.path = path
        self.labels = labels
        self.verbose = verbose
        self.clear()
        if filename is not None:
            self.load(filename)

    @staticmethod
    def _read_directory(path):
        """Read directory content"""
        fnlist = utl.scandir(path, "*.bmp")
        database = defaultdict(list)
        for fn in fnlist:
            fnsplit = fn.split('\\')
            database[fnsplit[-2]].append(fnsplit[-1])
        return database

    def _statistics(self, dataset, text):
        """Print data statistics"""
        if self.verbose:
            print(text)
            for key, val in dataset.items():
                print(f'{key} : {len(val)}')

    def create_splits(self, siztrn=0.6, sizval=0.4, siztst=0.0, shuffle=False, equalize=False):
        """Create trn/val/tst data splits"""

        # read data directory
        if self.verbose:
            print('Reading data directory ...')
        database = self._read_directory(self.path)
        self._statistics(database, 'Directory content:')

        # optionally equalize number of samples
        min_samples = math.inf
        if equalize:
            # find min number of samples across all labels
            for fnlist in database.values():
                min_samples = min(len(fnlist), min_samples)

        # create splits
        if  self.verbose:
            print('Creating trn/val/tst splits ...')
        self.clear()
        for label, fnlist in database.items():
            split = self._split_trnvaltst(fnlist, siztrn, sizval, siztst, shuffle, min_samples)
            for key, val in split.items():
                self.data[key][label] = val

        # print statistics
        for key, data in self.data.items():
            self._statistics(data, f'Set {key}:')

    def _split_trnvaltst(self, fnlist, siztrn, sizval, siztst, shuffle, maxitems):
        if shuffle:
            random.shuffle(fnlist)
        if math.isinf(maxitems):
            maxitems = len(fnlist)
        a = int(siztrn*maxitems)
        b = a + int(sizval*maxitems)
        c = b + int(siztst*maxitems)
        return {
            'trn': fnlist[0:a],
            'val': fnlist[a:b],
            'tst': fnlist[b:c]}

    def clear(self):
        """Clear data set splits"""
        self.data = {
            'trn' : defaultdict(list),
            'val' : defaultdict(list),
            'tst' : defaultdict(list)
        }

    def save(self, filename):
        """Save text file with data splits"""
        if self.verbose:
            print('Saving data splits ...')
        with open(filename, 'w') as fw:
            for type, data in self.data.items():
                for label, fnlist in data.items():
                    for fn in fnlist:
                        fw.write(f'{type},{label},{fn}\n')

    def load(self, filename):
        """Load data splits based on text file"""
        if self.verbose:
            print('Loading data splits ...')
        self.clear()
        # load
        with open(filename, 'r') as fr:
            for line in fr:
                tok = line.strip().split(',')
                self.data[tok[0]][tok[1]].append(tok[2])

        # print statistics
        for key, data in self.data.items():
            if len(data) is not 0:
                self._statistics(data, f'Set {key}:')

    def get_label_filename(self, setname):
        """Generator (label, path), where name = trn/val/tst"""
        for label, fnlist in self.data[setname].items():
            for fn in fnlist:
                yield label, os.path.join(self.path, label, fn)

    def get_label_image(self, setname):
        """Generator (label, image), where setname = trn/val/tst"""
        for label, fn in self.get_label_filename(setname):
            yield self.labels[label], Image.open(fn)

