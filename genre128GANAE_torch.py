# Genre128GANAE.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
import numpy as np
from data_loader import get_dataloaders
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from scipy.stats import entropy

class Generator(nn.Module):
    def __init__(self, zdim=100, n_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        self.main = nn.Sequential(
            # (zdim + n_classes) x 1 x 1
            nn.ConvTranspose2d(zdim + n_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, c], 1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 3 x 64 x 64
            nn.Conv2d(3, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.cls = nn.Linear(1024*4*4, n_classes)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 3, 3, 2, 1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(-1, 1024*4*4)
        cls_output = self.cls(features)
        recon = self.decoder(features.view(-1, 1024, 4, 4))
        return cls_output, recon
    

def calculate_inception_score(images, batch_size=32, splits=10):
    """计算Inception Score (IS)"""
    device = images.device
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            # 调整图像尺寸为Inception V3的输入要求（299x299）
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear')
            pred = model(batch)
            preds.append(torch.softmax(pred, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    scores = []
    for i in range(splits):
        split = preds[i * len(preds)//splits : (i+1)*len(preds)//splits]
        p_y = np.mean(split, axis=0)
        kl = split * (np.log(split) - np.log(p_y))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)

def calculate_fid(real_images, fake_images, batch_size=50):
    """计算Fréchet Inception Distance (FID)"""
    device = real_images.device
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # 获取特征层输出
    model.eval()
    
    def get_features(imgs):
        features = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i:i+batch_size]
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear')
                feat = model(batch).cpu().numpy()
                features.append(feat)
        return np.concatenate(features, axis=0)
    
    real_feat = get_features(real_images)
    fake_feat = get_features(fake_images)
    
    mu1, sigma1 = np.mean(real_feat, axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = np.mean(fake_feat, axis=0), np.cov(fake_feat, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0].real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.gen_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    netG = Generator(zdim=args.zdim, n_classes=args.n_classes).to(device)
    netD = Discriminator(n_classes=args.n_classes).to(device)
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    train_loader, val_loader = get_dataloaders(args.batch_size, args.img_size)
    
    fixed_z = torch.randn(64, args.zdim, 1, 1, device=device)
    fixed_labels = torch.randint(0, args.n_classes, (64,), device=device)
    
    for epoch in range(args.epochs):
        for i, (real_imgs, real_labels) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_imgs.size(0)
            
            netD.zero_grad()
            
            real_cls, real_recon = netD(real_imgs)
            errD_real_cls = nn.CrossEntropyLoss()(real_cls, real_labels)
            errD_real_recon = torch.mean((real_recon - real_imgs)**2)
            
            z = torch.randn(batch_size, args.zdim, 1, 1, device=device)
            fake_labels = torch.randint(0, args.n_classes, (batch_size,), device=device)
            fake_imgs = netG(z, fake_labels)
            fake_cls, fake_recon = netD(fake_imgs.detach())
            errD_fake_cls = nn.CrossEntropyLoss()(fake_cls, fake_labels)
            
            errD = errD_real_cls + 0.5*errD_real_recon + errD_fake_cls
            errD.backward()
            optimizerD.step()
            
            netG.zero_grad()
            fake_cls, fake_recon = netD(fake_imgs)
            errG_cls = nn.CrossEntropyLoss()(fake_cls, fake_labels)
            errG_recon = torch.mean((fake_recon - fake_imgs)**2)
            errG = errG_cls + 0.5*errG_recon
            errG.backward()
            optimizerG.step()
            
            if i % 100 == 0:
                print(f"[{epoch}/{args.epochs}][{i}/{len(train_loader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")
                
            if i % args.save_interval == 0:
                with torch.no_grad():
                    fake = netG(fixed_z, fixed_labels)
                    save_image(fake, os.path.join(args.gen_dir, f"epoch_{epoch}_iter_{i}.png"), 
                               normalize=True, nrow=8)
                    
                    # eval_z = torch.randn(1000, args.zdim, 1, 1, device=device)
                    # eval_labels = torch.randint(0, args.n_classes, (1000,), device=device)
                    # fake_images = netG(eval_z, eval_labels)
                    
                    real_images, _ = next(iter(val_loader))
                    real_images = real_images.to(device)
                    
                    is_mean, is_std = calculate_inception_score(fake)
                    print(f"[Epoch {epoch}] IS: {is_mean:.2f} ± {is_std:.2f}")
                    
                    fid_score = calculate_fid(real_images, fake)
                    print(f"[Epoch {epoch}] FID: {fid_score:.2f}")
        
        if epoch % args.save_epoch == 0:
            torch.save(netG.state_dict(), os.path.join(args.model_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(args.model_dir, f"netD_epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--zdim', type=int, default=100)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_epoch', type=int, default=1000)
    parser.add_argument('--gen_dir', default='./genimgs/Genre128GANAE')
    parser.add_argument('--model_dir', default='./models/Genre128GANAE')
    
    args = parser.parse_args()
    
    train(args)