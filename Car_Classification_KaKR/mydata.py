import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import glob

class mydataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.image_list = glob.glob(data_path + 'train/*.jpg')
        self.target_list = pd.read_csv(data_path + 'train.csv')
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])
        target = self.target_list.iloc[idx, 5]

        if self.transform:
            img = self.transform(img)

        return img, target
        