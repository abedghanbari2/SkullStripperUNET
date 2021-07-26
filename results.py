#%%
import torch
import numpy as np
from dataset import load_data
from metrics import dice_score
from torchvision import transforms
from PIL import Image
from utils import cvt1to3channels, normalize_image
import matplotlib.pyplot as plt


def get_dice_val(model, src_val, msk_val, fnames_val, trans):
    max_slice_num = max([i[1] for i in fnames_val])
    dice_val = {}

    for i_slice in range(max_slice_num):
        print(f'Starting slice number: {i_slice}')
        dice_val[i_slice] = []
        idx_slice = [i for i, f in enumerate(fnames_val) if f[1]==i_slice]
        inputs_all = torch.empty([len(idx_slice),3,256,256])
        labels_all = torch.empty([len(idx_slice),1,256,256])
        for ii, idx1 in enumerate(idx_slice):
            inputs, labels = src_val[idx1], msk_val[idx1]
            inputs = trans(Image.fromarray(cvt1to3channels(inputs)))
            labels = trans(Image.fromarray(np.uint8(normalize_image(labels))))
            inputs_all[ii,:,:,:] = inputs
            labels_all[ii,:,:,:] = labels
        
        model.eval()
        with torch.no_grad():
            predict = model(inputs_all)

        # Saving metrics
        dice_val[i_slice] = dice_score(labels_all, predict).item()
        print(f'Validation dice score for slice number: {i_slice} of {modality} was {dice_val[i_slice]}')
    return dice_val

dice_modality = {}
modalities = ['dti', 'fmri', 'anatomical', 'noddi']
for modality in modalities:
    skpath='/home/ghanba/ssNET/SkullStripperUNET/weights/'+modality+'.pth'
    # skpath='/home/ghanba/ssNET/skull-stripper/paper_weights/skull-stripper-paper.pth'
    data_path = '/projects/compsci/USERS/frohoz/msUNET/train/dataset/'
    validation_portion=.2
    _, _, src_val, msk_val, fnames_train, fnames_val = load_data(data_path,
    num_train_sample=80, modality=modality)
    Nt = len(_) # number of trian samples

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                        in_channels=3, out_channels=1, init_features=32,
                        pretrained=False)
    model.load_state_dict(torch.load(skpath, map_location=torch.device('cpu')))

    trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])
    print(f'Calculating dice score for slices in {modality}')
    dice_modality[modality] = get_dice_val(model, src_val, msk_val, fnames_val, trans)

#%%
for modality in modalities:
    plt.plot(dice_modality[modality].values(),'.')

plt.xlabel('Slice Number')
plt.ylabel('Mean Dice Coeff')
plt.legend(modalities)
plt.ylim((.85, 1))
plt.savefig('results_.pdf')
plt.show()
# %%
i_slice = 7
idx_slice = [i for i, f in enumerate(fnames_val) if f[1]==i_slice]

for idx1 in idx_slice[:10]:
    inputs, labels = src_val[idx1], msk_val[idx1]
    labels = trans(Image.fromarray(np.uint8(normalize_image(labels))))
    # predict = model(trans(Image.fromarray(cvt1to3channels(inputs))).unsqueeze(0))
    model.eval()
    with torch.no_grad():
        predict = model(trans(Image.fromarray(cvt1to3channels(inputs))).unsqueeze(0))
    print(f'dice score: {dice_score(predict[0,0,:,:], labels[0,:,:])}')

    plt.subplot(1,2,1)
    plt.title(f'max of labels: {np.max(labels[0,:,:].numpy())}')
    plt.imshow(labels[0,:,:])

    plt.subplot(1,2,2)
    plt.title(f'max of predict: {np.max(predict[0,0,:,:].detach().numpy())}')
    plt.imshow(predict.detach().numpy()[0,0,:,:])
    plt.show()

# %%
