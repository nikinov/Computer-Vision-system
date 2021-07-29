"""
File utils

June 2020, Pavel Krizek
"""

import os
import glob
import sys
import json
import csv
import pickle


def mkdir(dirname):
    "Create directory"
    os.makedirs(dirname, exist_ok=True)
    return os.path.exists(dirname)


def scandir(dirname, ext='*.png'):
    "Scan directory for files with a given extension"
    return glob.glob(os.path.join(dirname, '**', ext), recursive=True)


def splitfilename(filename):
    "Return path, filename, ext"
    path, fn = os.path.split(filename)
    fn, ext = os.path.splitext(fn)
    return path, fn, ext


def load_json(filename):
    "Load json file as a dictionary"
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data


def save_json(filename, data):
    "Save data (dictionary) to a json file"
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_csv(file_name, delimiter=','):
    """Load CSV file as a list with a dictionary"""
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)
        data = [row for row in reader]
    return data


def save_csv(filename, data, delimiter=','):
    "Save data (list with a dictionary) to a csv file"
    with open(filename, mode='w', newline='') as csv_file:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def load_list(filename):
    """Load text file line by line"""
    with open(filename, 'r') as fr:
        rows = fr.read().splitlines()
    return rows


def save_list(filename, thelist):
    """Save list line by line to a text file"""
    with open(filename, 'w') as fw:
        for item in thelist:
            fw.write(f'{item}\n')


def save_pickle(filename, data):
    """Save data to pickle file"""
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """Save data from pickle file"""
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def save_cmdline_args(filename):
    """Save command line arguments"""
    with open(filename, 'w') as f:
        f.write(' '.join(sys.argv))


def compare_files(file1, file2, buffer_size=512):
    """Compare content of two files

        :returns: True if similar, False if different and message
    """

    if not os.path.isfile(file1) or not os.path.isfile(file2):
        return False, "not found"

    if os.path.getsize(file1) != os.path.getsize(file2):
        return False, "size"

    f1 = open(file1, 'rb')
    f2 = open(file2, 'rb')

    result = True
    while result:
        buffer1 = f1.read(buffer_size)
        buffer2 = f2.read(buffer_size)
        if len(buffer1) == 0 or len(buffer2) == 0:
            break
        for b1, b2 in zip(buffer1, buffer2):
            if b1 != b2:
                result = False
                break

    f1.close()
    f2.close()

    return result, 'content' if result is False else 'identical'
