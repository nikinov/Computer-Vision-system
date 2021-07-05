
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure


img = img_as_float(io.imread("images/Alloy_noisy.jpg"))

sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=5, patch_distance=3, multichannel=True)


eq_img = exposure.equalize_adapthist(denoise_img)