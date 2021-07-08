import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("assets/img.png")
kernel = np.ones((3,3), np.float)/9
filt_2D = cv.filter2D(img, -1, kernel)
blur = cv.blur(img, (3, 3))
gaussian_blur = cv.GaussianBlur(img, (3,3), 0)
median_blur = cv.medianBlur(img, 3)
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)


cv.imshow("original", img)
cv.imshow("custom filter", filt_2D)
cv.imshow("blur", blur)
cv.imshow("median blur", median_blur)
cv.imshow("bilateral", median_blur)
cv.waitKey(0)
cv.destroyWindow()

