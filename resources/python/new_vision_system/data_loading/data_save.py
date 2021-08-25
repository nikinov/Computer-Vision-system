import os
import csv
import glob


class csv_save():
    def __init__(self, file_save_name, data_dir, train_split=0.8):
        # important double check the directory separator
        self.separator = "\\"
        self.class_track = {}
        self.train = []
        self.val = []
        for entry in glob.iglob(data_dir + '/**/*.bmp', recursive=True):
            temp = entry.split(self.separator)[-2]
            if temp in self.class_track.keys():
                self.class_track[temp].append(entry)
            else:
                self.class_track[temp] = []

        self.smallest_class_num = len(list(self.class_track.values())[0])
        for n in list(self.class_track.values()):
            self.smallest_class_num = min(self.smallest_class_num, len(n))
        for ls in list(self.class_track.values()):
            for i in range(len(ls) - self.smallest_class_num):
                ls.pop()
            for i, el in enumerate(ls):
                if i < self.smallest_class_num * train_split:
                    self.train.append(el)
                elif i > self.smallest_class_num * train_split:
                    self.val.append(el)

        # open knew csv file
        f = open("../csv/" + file_save_name + ".csv", "w")

        # create the csv writer
        writer = csv.writer(f)

        for im in self.train:
            writer.writerows(["trn", str(int(list(self.class_track.keys()).index(im.split(self.separator)[-2]))), im])
        for im in self.train:
            writer.writerows(["val", str(int(list(self.class_track.keys()).index(im.split(self.separator)[-2]))), im])
        f.close()


