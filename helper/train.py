import os
import torch
import data_setup, engine, train, utils
from torchvision import transforms



NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001



train_dir = "data/data_name/train"
test_dir = "data/data_name/test"

device="cuda" if torch.cuda.is_available() else "cpu"

data_transforms = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir,data_transforms=data_transforms,batch_size= BATCH_SIZE)

#Model Builder implementation remaining


