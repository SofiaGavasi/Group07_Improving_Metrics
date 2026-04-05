from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CIFAR10FIDDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data = data_dict[b'data']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img
