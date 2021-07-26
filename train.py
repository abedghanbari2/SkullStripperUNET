# Train skull stripper UNET using new data
import os
import torch; torch.cuda.empty_cache()
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import numpy as np
from dataset import SkullStripperDataset, load_data
from loss import DiceLoss
from metrics import dice_score
import argparse

def train_skullstripper(data_path,
                        num_train_sample,
                        batch_size=256,
                        lr=0.01,
                        lr_step=100,
                        modality=None,
                        num_epochs=100,
                        skpath=None,
                        save_path='weights'
                        ):
    
    print(f'Starting training: lr: {lr}, batch_size: {batch_size}, modality: {modality}, number of mice samples: {num_train_sample}')

    # where to save the model
    modelpath = os.path.join(save_path, modality+'.pth')

    # Load the data
    src_train, msk_train, src_val, msk_val, _, _ = load_data(data_path, num_train_sample, modality=modality)

    # Transform both images and masks ...
    trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])

    train = SkullStripperDataset(src_train, msk_train, transform=trans, augmentation=True)
    val = SkullStripperDataset(src_val, msk_val, transform=trans, augmentation=False)

    training = DataLoader(train, batch_size=batch_size, shuffle=False)
    validating = DataLoader(val, batch_size=batch_size, shuffle=False)

    # Load UNET model and weights from skull-stripper paper
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                        in_channels=3, out_channels=1, init_features=32,
                        pretrained=False)
    model.load_state_dict(torch.load(skpath, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()

    loss_f = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)

    # Metric placeholders
    train_loss, val_loss, train_dice_scores, val_dice_scores = [], [], [], []
    best_dice_score = 0

    # Train the model
    for epoch in range(num_epochs):

        # Train
        model.train()
        l_temp, train_dice_score_temp = [], []
        for inputs, labels in training:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            predict = model(inputs)

            loss = loss_f(predict, labels)
            
            loss.backward()
            optimizer.step()

            # Saving metrics
            l_temp.append(loss.item())
            train_dice_score_temp.append(dice_score(labels, predict).item())
        
        train_loss.append(l_temp)
        train_dice_scores.append(train_dice_score_temp)

        # scheduler.step()

        # Validation
        model.eval()
        l_temp, val_dice_score_temp = [], []
        for inputs, labels in validating:
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                predict = model(inputs)
                loss = loss_f(predict, labels)

                # Saving metrics
                l_temp.append(loss.item())
                val_dice_score_temp.append(dice_score(labels, predict).item())
            
        val_loss.append(l_temp)
        val_dice_scores.append(val_dice_score_temp)
            
        print('epoch [{}/{}], loss train:{:.4f}, val:{:.4f} || dice score train: {:.4f}, val: {:.4f}'.format(epoch+1, num_epochs, \
                    np.mean(train_loss[-1]), np.mean(val_loss[-1]), \
                    np.mean(train_dice_scores[-1]), np.mean(val_dice_scores[-1])), \
                )
        # Saving trained model
        if best_dice_score < np.mean(val_dice_scores[-1]):
            best_dice_score = np.mean(val_dice_scores[-1])
            torch.save(model.state_dict(), modelpath)

    return train_loss, val_loss, train_dice_scores, val_dice_scores

if __name__ == "__main__":
    data_path = '/projects/compsci/USERS/frohoz/msUNET/train/dataset/'
    parser=argparse.ArgumentParser()
    parser.add_argument("-m", "--modality", type=str, required=False)
    parser.add_argument("-n", "--num_train_sample", type=int, required=True)
    args=parser.parse_args()
    print(f'Starting training for modality {args.modality}')
    train_skullstripper(data_path,
                        num_train_sample=args.num_train_sample,
                        batch_size=64,
                        lr=0.01,
                        lr_step=100,
                        modality=args.modality,
                        num_epochs=100,
                        # skpath='/home/ghanba/ssNET/SkullStripperUNET/weights/'+args.modality+'.pth',
                        skpath='/home/ghanba/ssNET/skull-stripper/paper_weights/skull-stripper-paper.pth'
                        )
