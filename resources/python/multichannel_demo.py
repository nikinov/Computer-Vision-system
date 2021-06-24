import torch
import cv2 as cv
import numpy as np
import os
from torchvision import transforms

path = "otherImages/_Bad_Void"

image = cv.imread(os.path.join(path, "01055538070_testfield_1028_2.bmp"))
image2 = cv.imread(os.path.join(path, "01055542031_testfield_1028_6.bmp"))

b,g,r = cv.split(image)
b2,g2,r2 = cv.split(image2)

col = np.vstack([r, g, b])
col2 = np.vstack([r2, g2, b2])

full = np.hstack([col, col2])

im = cv.resize(full, (299, 299))

cv.imshow("image", im)
cv.waitKey()
























