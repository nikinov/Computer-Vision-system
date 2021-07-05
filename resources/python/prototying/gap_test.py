import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
from skimage.filters import threshold_otsu

import glob



img = io.imread("filters/assets/scratch.jpeg")
entropy_img = entropy(img, disk(10))
thresh = threshold_otsu(entropy_img)
binary = entropy_img <= thresh
print(np.sum(binary == True))

