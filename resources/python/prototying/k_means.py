import numpy as np
import cv2

img = cv2.imread("assets/test_void.bmp")
or_shape = img.shape

img = np.float32(img.reshape((-1, 3)))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k=2

attemts=10

ret, label, center = cv2.kmeans(img, k, None, criteria, attemts, cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
res = res.reshape((or_shape))
cv2.imshow("segmented", res)

cv2.waitKey(0)