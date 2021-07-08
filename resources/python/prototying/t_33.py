import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, color, io


img = cv2.imread("assets/img_3.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pixels_to_um = 0.5

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

sure_bg = cv2.dilate(opening, kernel, iterations=1)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

re2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
markers+=10

markers[unknown==255] = 0

plt.imshow(markers, cmap="jet")
plt.show()
#cv2.imshow("unknown pixels", unknown)
#cv2.waitKey()

