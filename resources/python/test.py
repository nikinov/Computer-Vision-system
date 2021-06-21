from PIL import Image
import glob
import os

out_dir = 'pngImages/'
cnt = 0
for img in glob.glob('otherImages/_Bad_Void/*.bmp'):
    Image.open(img).save(os.path.join(out_dir, str(cnt) + '.png'))
    cnt += 1