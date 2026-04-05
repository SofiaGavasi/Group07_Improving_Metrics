import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

class GANDataLoader:
    def __init__(self, generator, latent_dim, batch_size, num_batches, device):
        self.generator = generator
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1,3,1,1)
   
    def __iter__(self):
        self.generator.eval()
        with torch.no_grad():
            for _ in range(self.num_batches):
                z = torch.randn(self.batch_size, self.latent_dim, 1, 1).to(self.device)  # DCGAN expects [B, Z, 1, 1]
                imgs = self.generator(z)                    # [B, 3, 32, 32], range [-1, 1]

                imgs = (imgs + 1) / 2                       # → [0, 1]
                imgs = F.interpolate(imgs, size=(299, 299), # → [B, 3, 299, 299]
                                    mode='bilinear', 
                                    align_corners=False)

                mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1,3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1,3,1,1)
                imgs = (imgs - mean) / std                  # ImageNet normalize

                yield imgs

    def __len__(self):
        return self.num_batches