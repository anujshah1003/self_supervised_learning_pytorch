import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
#        return np.flipud(np.transpose(img, (1, 0, 2)))
        return img.rotate(90)
    elif rot == 180:  # 90 degrees rotation
#        return np.fliplr(np.flipud(img))
        return img.rotate(180)
    elif rot == 270:  # 270 degrees rotation / or -90
#        return np.transpose(np.flipud(img), (1, 0, 2))
        return img.rotate(270)
    elif rot == 120:
#        return ndimage.rotate(img, 120, reshape=False)
        return img.rotate(120)
    elif rot == 240:
#        return ndimage.rotate(img, 240, reshape=False)
        return img.rotate(240)
    else:
        raise ValueError('rotation should be 0, 90, 120, 180, 240 or 270 degrees')

