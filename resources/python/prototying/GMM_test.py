import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

img = cv2.imread("assets/test_void.bmp")

# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

# covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

original_shape = img.shape
segmented = gmm_labels.reshape(original_shape[0], original_shape[1])
plt.imshow(segmented)
plt.show()
