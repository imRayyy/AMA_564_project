# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import os

class WikiArtDataset(Dataset):
    def __init__(self, manifest_path, root_dir, transform=None, img_size=64):
        self.manifest = pd.read_csv(manifest_path, header=None)
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.label_dict = {i: torch.tensor(i) for i in range(10)}  

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        img_relpath, label_relpath = self.manifest.iloc[idx]
        img_path = os.path.join(self.root_dir, img_relpath)
        
        try:
            image = Image.open(img_path).convert('RGB')
            label = int(os.path.basename(label_relpath).split('.')[0])
            
            # 强制转换尺寸
            if image.size != (self.img_size, self.img_size):
                image = self.transform(image)
            else:
                image = self.transform(image)
                
            return image, torch.tensor(label)
        except Exception as e:
            print(f"加载 {img_path} 失败: {str(e)}")
            return torch.zeros(3, self.img_size, self.img_size), torch.tensor(-1)
        

def get_dataloaders(batch_size=100, img_size=64):
    train_transform = transforms.Compose([
        transforms.Resize(img_size),          
        transforms.CenterCrop(img_size),      
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),    
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = WikiArtDataset(
        manifest_path='./dataset/wikiart/genre-train-index.csv',
        root_dir='./dataset/wikiart',
        transform=train_transform
    )
    
    val_set = WikiArtDataset(
        manifest_path='./dataset/wikiart/genre-val-index.csv',
        root_dir='./dataset/wikiart',
        transform=val_transform
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader