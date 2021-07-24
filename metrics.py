def dice_score(input, target):
    '''
    input and target: Tensors
    '''
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.*intersection+smooth) / (iflat.sum() + tflat.sum() + smooth)
