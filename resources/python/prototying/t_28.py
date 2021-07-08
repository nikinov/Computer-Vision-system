# erosian
# dialation


import cv2 as cv
import numpy as np

img = cv.imread("assets/img_2.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

#harris = cv.cornerHarris()


# tutorial:
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html