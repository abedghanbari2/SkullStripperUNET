import numpy as np


def cvt1to3channels(one_channel):
    return np.stack((one_channel,)*3, axis=-1)

def normalize_image(image):
    return 255*((image - np.min(image)) / (np.max(image) - np.min(image)))

def dice_score(input, target):
    '''
    input and target are Tensors
    '''
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.*intersection+smooth) / (iflat.sum() + tflat.sum() + smooth)
