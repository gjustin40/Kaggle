import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import glob
from PIL import Image
import numpy as np
import os

class sub_data(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.image_list = glob.glob(self.data_path + 'test/*.jpg')

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx])

        if self.transform:
            img = self.transform(img)

        return img

def main():

    print('===Data Preparing...')
    data_path = '../../data/car_classification/'
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    dataset = sub_data(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    print(len(dataloader))

    print('===Cuda Checking...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)


    print('===Building Model...')
    model = models.vgg16(pretrained=False)
    model = model.to(device)
    
    for p in model.parameters():
        print(p['weight'])
        p.requires_grad = False
    
    model.classifier[6].out_features = 196
    
    checkpoint = torch.load('checkpoint/ckpt_vgg.pth')
    model.load_state_dict(checkpoint['model'])
    print(model.parameters())
    

    # prediction
    print('===Predicting...')
    result = []
    for i, images in enumerate(dataloader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        result += predicted.cpu().tolist()

        if i % 50 == 0:
            print('진행중.....[%d/%d]' % (i, len(dataloader)))

    print('진행중.....[%d/%d]' % (len(dataloader), len(dataloader)))

    if not os.path.isdir('result'):
        os.mkdir('./result')
    torch.save(result, './result/result.pkl')



if __name__ == '__main__':
    main()

