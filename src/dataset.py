import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self):
        self.file_paths = []
        for style_directory in os.scandir('../data/images'):
            sub_path = os.path.join('../data/images', style_directory)
            for artist_directory in os.scandir(sub_path):   
                path = os.path.join(sub_path, artist_directory)
                file_paths += [os.path.join(path, f) for f in os.scandir(path)]

        artists = [f.split('/')[-2] for f in self.file_paths]
        artists = list(set(artists))
        self.artist_to_index = {artist: artists.index(artist) for artist in artists}   

        styles = [f.split('/')[-3] for f in self.file_paths]
        styles = list(set(styles))
        self.style_to_index = {style: styles.index(style) for style in styles}

    def __len__(self):
        return len(self.filePaths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = read_image(img_path)

        artist_label = img_path.split('/')[-2]
        artist_label = self.artist_to_index[artist_label]

        artstyle_label = img_path.split('/')[-3]
        self.artstyle_label = self.style_to_index[artist_label]
        
        return image, artstyle_label, artist_label