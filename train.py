# Train skull stripper UNET using new data
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import optim
from torchvision import transforms
from dataset import SkullStripperDataset, load_data
from loss import DiceLoss

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
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)