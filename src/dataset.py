import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self):
        self.file_paths = []
        for directory in os.scandir('../data/images'):
            path = os.path.join('../data/images', directory)
            file_paths += [os.path.join(path, f) for f in os.scandir(path)]

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = read_image(img_path)
        artstyle_label = img_path.split('/')[-2]
        
        return image, artstyle_label