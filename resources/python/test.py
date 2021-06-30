from PIL import Image
import glob
import os
import cv2
import numpy as np
"""
out_dir = 'pngImages/'
cnt = 0
for img in glob.glob('otherImages/_Bad_Void/*.bmp'):
    Image.open(img).save(os.path.join(out_dir, str(cnt) + '.png'))
    cnt += 1"""

dirs = "../AssetsGray/_bad_brown"
i = 0
for root, dirs, files in os.walk(dirs):
    for file in files:
        if not file.endswith(".DS_Store") and file.endswith(".bmp"):
            pt = os.path.join(root, file)
            im = cv2.imread(pt)
            r, g, b = cv2.split(im)
            for j in range(2):
                cv2.imwrite(os.path.join(root, str(i) + "r.bmp"), r)
                cv2.imwrite(os.path.join(root, str(i) + "g.bmp"), g)
                cv2.imwrite(os.path.join(root, str(i) + "b.bmp"), b)
                i += 1
                print(j)
            os.remove(pt)


