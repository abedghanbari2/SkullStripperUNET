import numpy as np


def cvt1to3channels(one_channel):
    return np.stack((one_channel,)*3, axis=-1)

def normalize_image(image):
    return 255*((image - np.min(image)) / (np.max(image) - np.min(image)))

