import torchvision

inception = torchvision.models.inception_v3(pretrained=True)
inception.eval()

import math
import random

import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import cv2
class CocoDataset(Dataset):

    def __init__(self, dir_, transform = None, size_h=299, size_w=299):


        self.size_h = size_h
        self.size_w = size_w
        self.transform = transform
        self.dataset = []
        for file in os.listdir(dir_):
            self.dataset.append(os.path.join(dir_, file))
        

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):

        image = cv2.imread(self.dataset[idx])

        image = cv2.resize(image, (self.size_h, self.size_w))
 
        if self.transform:
            image = self.transform(image)

        return image



train_dataset = CocoDataset('./coco/train2014/',
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                 ]))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)


class Iden(nn.Module):
    def __init__(self):
        super(Iden, self).__init__()
        
    def forward(self, x):
        return x

inception.fc = Iden().cuda()



list_ = []
for en, batch in enumerate(train_loader, 0):

    out = inception(batch)
    list_.append(out.cpu().detach().numpy())
        
    if en%16==0:

        np.save('save/embed_'+str(en), np.concatenate(list_, axis = 0))
        list_ = []
