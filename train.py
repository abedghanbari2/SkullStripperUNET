# Train skull stripper UNET using new data
import os
import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import numpy as np
from dataset import SkullStripperDataset, load_data
from loss import DiceLoss
from metrics import dice_score

def train_skullstripper(data_path,
                        validation_portion=.2,
                        batch_size=256,
                        lr=0.01,
                        num_epochs=100,
                        skpath='/home/ghanba/ssNET/skull-stripper',
                        ):
    # Load the data
    src_train, msk_train, src_val, msk_val = load_data(data_path, validation_portion=validation_portion)

    # Transform both images and masks ...
    trans = transforms.Compose([transforms.Resize((225,225)),transforms.CenterCrop(256), transforms.ToTensor()])

    train = SkullStripperDataset(src_train, msk_train, transform=trans)
    val = SkullStripperDataset(src_val, msk_val, transform=trans)

    training = DataLoader(train, batch_size=batch_size, shuffle=False)
    validating = DataLoader(val, batch_size=batch_size, shuffle=False)

    # Load UNET model and weights from skull-stripper paper
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                        in_channels=3, out_channels=1, init_features=32,
                        pretrained=False)
    model.load_state_dict(torch.load(os.path.join(skpath,"paper_weights/skull-stripper-paper.pth"),map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        model.cuda()

    loss_f = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metric placeholders
    train_loss, val_loss, train_dice_scores, val_dice_scores = [], [], [], []

    # Train the model
    for epoch in range(num_epochs):

        # Train
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

        # Validation
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
        
    return train_loss, val_loss, train_dice_scores, val_dice_scores

if __name__ == "__main__":
    data_path = '/projects/compsci/USERS/frohoz/msUNET/train/dataset/'
    train_skullstripper(data_path,
                        validation_portion=.2,
                        batch_size=64,
                        lr=0.01,
                        num_epochs=100,
                        skpath='/home/ghanba/ssNET/skull-stripper',
                        )
