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
from random import seed, shuffle

def load_data(data_path, num_train_sample, modality=None):

    train_path = data_path + 'training/' 
    mask_path = data_path + 'masks/' 
    train_list = sorted(os.listdir(train_path))

    mouse_name = np.unique([f[:24] for f in train_list if f[:6]=='CK_DIF'])
    seed(10)
    shuffle(mouse_name)
    print(f'Number of unique mice found in dataset: {len(mouse_name)}')
    mouse_train = mouse_name[:num_train_sample]
    mouse_val   = mouse_name[num_train_sample:]

    train_list = [i for i in train_list if i[-2:]=='gz']

    src_train, msk_train, fnames_train = [], [], []
    src_val, msk_val, fnames_val = [], [], []

    # Read indiviual "nii.gz" files
    for f in train_list:
        if modality and modality not in f:
                continue
        # Read the entire volume
        training_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path + f))
        
        # 'noddi' images have 4 dimentions
        if len(training_arr.shape)==4:
            training_arr = training_arr[0,:,:,:]
        
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path + f.replace('data','mask')))

        for image_idx in range(training_arr.shape[0]):
            # Preprocess and transform training data
            input_image_original = training_arr[image_idx, :,:]
            input_image_original = normalize_image(input_image_original)

            # Transform expert mask
            input_mask_original = mask_arr[image_idx, :,:]
            input_mask_original = normalize_image(input_mask_original)
            if f[:24] in mouse_train:
                src_train.append(np.uint8(input_image_original))
                msk_train.append(np.uint8(input_mask_original))
                fnames_train.append((f, image_idx))
            elif f[:24] in mouse_val:
                src_val.append(np.uint8(input_image_original))
                msk_val.append(np.uint8(input_mask_original))
                fnames_val.append((f, image_idx))
            else:
                print(f'{f} is not in train or validation mouse list!')
            
    return src_train, msk_train, src_val, msk_val, fnames_train, fnames_val


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
            # if random.random() > 0.5 and self.augmentation:
            #     image = F.vflip(image)
            #     mask = F.vflip(mask)
            # if random.random() > 0.5 and self.augmentation:
            #     image = F.hflip(image)
            #     mask = F.hflip(mask)
            # if random.random() > 0.5 and self.augmentation:
            #     angle=np.random.choice([5.0,-5.0])
            #     image = F.rotate(image,angle)
            #     mask = F.rotate(mask,angle)

            image = self.transform(image)
            mask = self.transform(mask)


        return image, mask