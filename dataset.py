# Loading data and defining dataset class
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import numpy as np
import SimpleITK as sitk
from utils import cvt1to3channels, normalize_image
import random
from PIL import Image
from torchvision import transforms
import os


def load_data(data_path, validation_portion=0.2):
    '''
    loads all train images in data_path
    data_path has two subfolders "training" and "masks"
    images of all modalities are saved as .nii.gz format
    there should be a corresponding file in masks for each file in training
    #TODO check for this correspondence

    all frames from all modalities are stored in output list
    '''

    train_path = data_path + 'training/' 
    mask_path = data_path + 'masks/' 
    train_list = sorted(os.listdir(train_path))
    train_list = [i for i in train_list if i[-2:]=='gz']

    src = []
    msk = []

    # Read indiviual "nii.gz" files
    for f in train_list:
        if 'dti' in f:
            # Read the entire volume
            training_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path + f))
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path + f.replace('data','mask')))

            for image_idx in range(training_arr.shape[0]):
                # Preprocess and transform training data
                input_image_original = training_arr[image_idx, :,:]
                input_image_original = normalize_image(input_image_original)

                # Transform expert mask
                input_mask_original = mask_arr[image_idx, :,:]
                src.append(np.uint8(input_image_original))
                msk.append(np.uint8(input_mask_original))

    src = np.array(src)
    msk = np.array(msk)

    validation_size = int(len(src) * validation_portion)
    train_size = len(src)-validation_size

    return src[:train_size], msk[:train_size], \
            src[train_size:], msk[train_size:],


class SkullStripperDataset(Dataset):
    '''
    the training data is already includes augmentation from Zachary 
    '''
    def __init__(self, src, msk, 
                    transform=None,
                    augmentation=True):
        self.src = src
        self.msk = msk
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_img = self.src[idx]
        msk_img = self.msk[idx]
        
        image = cvt1to3channels(src_img)
        image = Image.fromarray(np.uint8(image))

        mask = Image.fromarray(np.uint8(msk_img))

        if self.transform:
            if random.random() > 0.5 and self.augmentation:
               image = F.vflip(image)
               mask = F.vflip(mask)
            if random.random() > 0.5 and self.augmentation:
               image = F.hflip(image)
               mask = F.hflip(mask)
            if random.random() > 0.5 and self.augmentation:
               angle=np.random.choice([90.0,180.0,270.0])
               image = F.rotate(image,angle)
               mask = F.rotate(mask,angle)

            image = self.transform(image)
            mask = self.transform(mask)


        return image, mask