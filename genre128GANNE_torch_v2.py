import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--init_iter', type=int, default=0)
parser.add_argument('--max_iter', type=int, default=50000)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--eval_iter', type=int, default=100)
parser.add_argument('--store_img_iter', type=int, default=1)
parser.add_argument('--save_iter', type=int, default=10)
parser.add_argument('--lr_init', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--zdim', type=int, default=100)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--im_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--data_root', type=str, default='./dataset/wikiart')
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--eval_num_samples', type=int, default=1000)

args = parser.parse_args()

os.makedirs('./genimgs/Genre128GANAE/gen', exist_ok=True)
os.makedirs('./genimgs/Genre128GANAE/real', exist_ok=True)
os.makedirs('./genimgs/Genre128GANAE/gen128', exist_ok=True)
os.makedirs('./models/Genre128GANAE', exist_ok=True)

class ArtDataset(Dataset):
    def __init__(self, manifest_path, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    parts = line.strip().rsplit(',', 1) 
                    if len(parts) != 2:
                        print(f"Skipping invalid line (分割错误) at line {line_idx}: {line.strip()}")
                        continue
                        
                    img_relpath, label_relpath = parts
                    
                    img_path = os.path.join(root_dir, img_relpath.strip())
                    label_path = os.path.join(root_dir, label_relpath.strip())
                    
                    if not os.path.exists(img_path):
                        print(f"Missing image at line {line_idx}: {img_path}")
                        continue
                        
                    if not os.path.exists(label_path):
                        print(f"Missing label at line {line_idx}: {label_path}")
                        continue
                        
                    with open(label_path, 'r', encoding='utf-8') as lf:
                        label_content = lf.read().strip()
                        try:
                            label = int(label_content)
                        except ValueError:
                            print(f"Invalid label format at line {line_idx}: {label_content}")
                            continue
                            
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    
                except Exception as e:
                    print(f"Error processing line {line_idx}: {str(e)}")
                    continue

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros(3, args.im_size, args.im_size), 0

transform = transforms.Compose([
    transforms.Resize((args.im_size, args.im_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = args.n_classes
        self.zdim = args.zdim
        
        # 初始全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.zdim + self.n_classes, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.LeakyReLU(0.2)
        )
        
        # 上采样模块
        self.deconv = nn.Sequential(
            # g2: 4x4 → 8x8
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # g3: 8x8 → 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # g4: 16x16 → 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # g5: 32x32 → 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # g5b: 64x64 → 64x64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # g6: 64x64 → 128x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # g6b: 输出层
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, y):
        y_onehot = F.one_hot(y, self.n_classes).float()
        z = torch.cat([z, y_onehot], dim=1)
        
        out = self.fc(z)
        out = out.view(-1, 512, 4, 4)  # 对应TF中的reshape到[512,4,4]
        
        out = self.deconv(out)
        
        out_64 = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)  
        return out_64, out  # (64x64, 128x128)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            # conv1: 64x64 → 32x32
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # conv2: 32x32 → 16x16
            nn.Dropout(args.dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # conv3: 16x16 → 8x8
            nn.Dropout(args.dropout),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # conv3b: 8x8 → 8x8
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # conv4: 8x8 → 4x4
            nn.Dropout(args.dropout),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        
        # 分类器
        self.classifier = nn.Linear(1024*4*4, args.n_classes)
        
        # 解码器部分
        self.decoder = nn.Sequential(
            # g1: 4x4 → 8x8
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # g2: 8x8 → 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # g3: 16x16 → 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # g4: 32x32 → 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # g5: 64x64 → 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 输出层
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        features = self.encoder(x)
        
        flat = features.view(features.size(0), -1)
        cls_pred = self.classifier(flat)
        
        recon = self.decoder(features)
        return cls_pred, recon

def log_sum_exp(x):
    m = torch.max(x, dim=1, keepdim=True)[0]
    return m + torch.log(torch.sum(torch.exp(x - m), dim=1, keepdim=True))


torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)

optimizer_G = optim.Adam(G.parameters(), lr=args.lr_init, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=args.lr_init, betas=(0.5, 0.999))

train_set = ArtDataset(
    os.path.join(args.data_root, 'genre-train-index.csv'),
    args.data_root,
    transform=transform
)
test_set = ArtDataset(
    os.path.join(args.data_root, 'genre-val-index.csv'),
    args.data_root,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size)

for iteration in range(args.init_iter, args.max_iter):
    lr = args.lr_init if iteration < 30000 else args.lr_init / 10
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr

    real_imgs, real_labels = next(iter(train_loader))
    # print("Input shape:", real_imgs.shape)
    real_imgs = real_imgs.to(device)
    real_labels = real_labels.to(device)
    
    optimizer_D.zero_grad()
    
    cls_pred_real, recon_real = D(real_imgs)

    # print("判别器输出验证:")
    # print("分类结果:", cls_pred_real.shape)  # [1, 10]
    # print("重建图像:", recon_real.shape) 
    
    z = torch.randn(args.batch_size, args.zdim, device=device)
    fake_imgs64, fake_imgs128 = G(z, real_labels)

    # print("Generator输出维度检查:")
    # print("64x64图像:", fake_imgs64.shape)  # 应输出 torch.Size([100, 3, 64, 64])
    # print("128x128图像:", fake_imgs128.shape)  # 应输出 torch.Size([100, 3, 128, 128])

    cls_pred_fake, recon_fake = D(fake_imgs64.detach())

    l_real = log_sum_exp(cls_pred_real)
    l_fake = log_sum_exp(cls_pred_fake)
    
    loss_D_real = -torch.mean(l_real) + torch.mean(F.softplus(l_real))
    loss_D_fake = torch.mean(F.softplus(l_fake))
    loss_cls_real = F.cross_entropy(cls_pred_real, real_labels)
    loss_recon_real = F.mse_loss(recon_real, real_imgs) * 0.5
    
    D_loss = loss_cls_real + loss_D_real + loss_D_fake + loss_recon_real
    
    D_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    
    _, recon_fake = D(fake_imgs64)
    cls_pred_fake, _ = D(fake_imgs64)
    
    l_fake = log_sum_exp(cls_pred_fake)
    loss_D_fake = -torch.mean(l_fake) + torch.mean(F.softplus(l_fake))
    loss_cls_fake = F.cross_entropy(cls_pred_fake, real_labels)
    loss_recon_fake = F.mse_loss(recon_fake, fake_imgs64) * 0.5
    
    G_loss = loss_D_fake + loss_cls_fake + loss_recon_fake
    G_loss.backward()
    optimizer_G.step()
    
    if iteration % args.display_iter == 0:
        print(f"Iter: {iteration} D_loss: {D_loss.item():.4f} G_loss: {G_loss.item():.4f}")
    
    if iteration % args.save_iter == 0:
        torch.save(G.state_dict(), f"./models/Genre128GANAE/G_{iteration}.pth")
        torch.save(D.state_dict(), f"./models/Genre128GANAE/D_{iteration}.pth")
    
    if iteration % args.store_img_iter == 0:
        with torch.no_grad():
            fake_64 = (fake_imgs64 + 1) / 2
            fake_128 = (fake_imgs128 + 1) / 2
            real = (real_imgs + 1) / 2
            
            save_image(fake_64, f"./genimgs/Genre128GANAE/gen/{iteration}.png")
            save_image(fake_128, f"./genimgs/Genre128GANAE/gen128/{iteration}.png")
            save_image(real, f"./genimgs/Genre128GANAE/real/{iteration}.png")