from PIL import Image
import pandas as pd

import torch.utils.data as D
import warnings
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
class CustomDatasets(D.Dataset):
    def __init__(self,path,data,transform=None):
        self.path = path
        self.data = pd.read_csv(data)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        name = self.data['name'].values[idx]
        image = Image.open(self.path+name)
        
        label = self.data['label'].values[idx]
        if self.transform:
            image = self.transform(image)
        return image,label