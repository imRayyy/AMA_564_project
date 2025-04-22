import torch
from torchvision.models import inception_v3
from scipy.stats import entropy
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy.linalg import sqrtm

def calculate_inception_score(images, batch_size=32, splits=10):
    model = inception_v3(pretrained=True)
    model.eval()
    model.to(device)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            pred = model(batch)
            preds.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i+1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores.append(entropy(py, np.mean(part, axis=1)))
    
    return np.exp(np.mean(scores))

def calculate_fid(real_images, fake_images, batch_size=50):
    model = inception_v3(pretrained=True)
    model.fc = torch.nn.Identity()  
    model.eval()
    model.to(device)

    def get_features(images):
        features = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                feat = model(batch)
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)
    
    real_feat = get_features(real_images)
    fake_feat = get_features(fake_images)
    
    mu1, sigma1 = np.mean(real_feat, axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = np.mean(fake_feat, axis=0), np.cov(fake_feat, rowvar=False)

    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


if __name__ ==  '__main__':

    real_loader = DataLoader(...)  
    fake_images = ...  

    is_score = calculate_inception_score(fake_images)
    print(f"Inception Score: {is_score:.2f}")

    real_images = next(iter(real_loader))[0].to(device)
    fid_score = calculate_fid(real_images, fake_images)
    print(f"FID Score: {fid_score:.2f}")
