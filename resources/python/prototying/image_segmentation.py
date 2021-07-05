from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte, io
import numpy as np
import matplotlib.pyplot as plt

img = img_as_float(io.imread("filters/assets/test_void.bmp"))

sigma_est = np.mean(estimate_sigma(img, multichannel=True))

denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False, patch_size=5, patch_distance=3, multichannel=True)

denoise_ubyte = img_as_ubyte(denoise)

seg1 = (denoise_ubyte <= 33)
seg2 = (denoise_ubyte > 33) & (denoise_ubyte <= 56)
seg3 = (denoise_ubyte > 56) & (denoise_ubyte <= 80)
seg4 = (denoise_ubyte > 80)

all_segmanets = np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3))

all_segmanets = int(seg1)
#all_segmanets[seg2] = (0, 1, 0)
#all_segmanets[seg3] = (0, 0, 1)
#all_segmanets[seg4] = (1, 1, 0)

plt.imshow(all_segmanets)
plt.show()

