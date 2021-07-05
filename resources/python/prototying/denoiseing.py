from skimage import io
from scipy import ndimage as nd
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

img = img_as_float(io.imread("filters/assets/test_void.bmp"))
gaussian_img = nd.gaussian_filter(img, sigma=5)
median_img = nd.median_filter(img, size=10)

##### NLM#####

sigma_est = np.mean(estimate_sigma(img, multichannel=True))

nlm = denoise_nl_means(img, h=1.5 * sigma_est, fast_mode=True, patch_size=10, multichannel=True)

plt.imshow(median_img)
#plt.imshow(img)
plt.show()



