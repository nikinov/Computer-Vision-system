import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


img = cv2.imread("assets/img_4.png")

img = np.float32(img.reshape((-1, 3)))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

attemts=10

n_components = np.arange(1, 10)

#kmeans_models = [cv2.kmeans(img, n, None, criteria, attemts, cv2.KMEANS_PP_CENTERS) for n in n_components]

gmm_model = [GMM(n, covariance_type='tied').fit(img) for n in n_components]

plt.plot(n_components, [m.bic(img) for m in gmm_model], label='BIC')

plt.show()