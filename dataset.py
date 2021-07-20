# Loading data and defining dataset class
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from utils import cvt1to3channels, normalize_image
from PIL import Image
from torchvision import transforms
import os


def load_data(data_path, validation_portion=0.05):
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
        # Read the entire volume
        training_arr = sitk.GetArrayFromImage(sitk.ReadImage(train_path + f))
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path + f.replace('data','mask')))

        for image_idx in range(training_arr.shape[0]):

            trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])

            # Preprocess and transform training data
            input_image_original = training_arr[image_idx, :,:]
            input_image_original = normalize_image(input_image_original)
            input_image = cvt1to3channels(input_image_original)
            input_image = Image.fromarray(np.uint8(input_image))
            input_image = trans(input_image).unsqueeze(0)
            
            # Transform expert mask
            input_mask_original = mask_arr[image_idx, :,:]
            input_mask = Image.fromarray(np.uint8(input_mask_original))
            input_mask = trans(input_mask).unsqueeze(0)

            src.append(np.uint8(input_image))
            msk.append(np.uint8(input_mask))


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
    def __init__(self, src, msk):
        self.src = src
        self.msk = msk

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.msk[idx]

if