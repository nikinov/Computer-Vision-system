import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("assets/img_1.png", 0)
img2 = cv.GaussianBlur(img, (3, 3), 0)

eq_img = cv.equalizeHist(img)

#cv.erode(th, kernel, iterations=1)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img)
cl_img2 = clahe.apply(img2)

plt.hist(eq_img.flat, bins=100, range=(0, 255))

ret1, thresh1 = cv.threshold(cl_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret2, thresh2 = cv.threshold(cl_img2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)


cv.imshow("original", img)
cv.imshow("thresh 1", thresh1)
cv.imshow("thresh 2", thresh2)
cv.waitKey()
