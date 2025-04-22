import torch
import argparse
import os
import numpy as np
from torchvision.utils import save_image
from genre128GANAE_torch import Generator  

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    netG = Generator(zdim=args.zdim, n_classes=args.n_classes).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    netG.load_state_dict(checkpoint)
    netG.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        z = torch.randn(args.num_images, args.zdim, 1, 1, device=device)
        labels = torch.randint(0, args.n_classes, (args.num_images,), device=device)
        
        fake_images = netG(z, labels)
        
        save_path = os.path.join(args.output_dir, "generated_images.png")
        save_image(fake_images, save_path, nrow=8, normalize=True)
        print(f"生成图像已保存至 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/Genre128GANAE/netG_epoch_0.pth',
                        help='生成器模型路径，例如 ./models/netG_epoch_50000.pth')
    parser.add_argument('--output_dir', type=str, default='./generated',
                        help='生成图像保存目录')
    parser.add_argument('--num_images', type=int, default=64,
                        help='生成图像数量')
    parser.add_argument('--zdim', type=int, default=100,
                        help='噪声维度，需与训练时一致')
    parser.add_argument('--n_classes', type=int, default=10,
                        help='类别数量，需与训练时一致')
    
    args = parser.parse_args()
    
    generate(args)