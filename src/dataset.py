import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torchvision
from torchvision import transforms


class ArtworkImageDataset(Dataset):
    def __init__(self, image_size, pair_by, pairing_scheme):
        """
        image_size: int ->
            Specifies the (image_size, image_size) resolution of the image

        pair_by: 'artist' or 'style' ->
            Specifies how we should group up images in the dataset. For example,
            pair_by = 'artist' will have the dataset generate pairs (x, y) of images
            with the same artist. For pair_by = 'style', the dataset will generate pairs
            (x, y) of images with the same art style

        pairing_scheme: 'positive' or 'negative' or 'both' ->
            Specifies the pairing scheme of images (x, y) in the dataset.
            Positive means we will only have pairs (x, y) with the same grouping
            (E.g. (x, y) have the same artist / artstyle, depending on the value of pair_by)
            Negative means we will only have pairs (x, y) with DIFFERENT groupings
            Both means we will have both positive and negative pairs

        """
        self.image_size = image_size

        self.file_paths = []
        for style_directory in os.scandir("../data/images"):
            sub_path = os.path.join("../data/images", style_directory)

            for artist_directory in os.scandir(sub_path):
                path = os.path.join(sub_path, artist_directory)
                file_paths += [os.path.join(path, f) for f in os.scandir(path)]

        artists = [f.split("/")[-2] for f in self.file_paths]
        artists = list(set(artists))
        self.artist_to_index = {artist: artists.index(artist) for artist in artists}
        self.index_to_artist = {idx: artist for artist, idx in self.artist_to_index}

        styles = [f.split("/")[-3] for f in self.file_paths]
        styles = list(set(styles))
        self.style_to_index = {style: styles.index(style) for style in styles}
        self.index_to_style = {idx: style for style, idx in self.style_to_index}

        self.pair_by = pair_by
        self.pairing_scheme = pairing_scheme
        self.pairings = self.initialize_pairings()

    def get_images_by_style(self):
        file_paths = {}

        for style_directory in os.scandir("../data/images"):
            file_paths[style_directory] = []
            sub_path = os.path.join("../data/images", style_directory)

            for artist_directory in os.scandir(sub_path):
                path = os.path.join(sub_path, artist_directory)
                file_paths[style_directory] += [
                    os.path.join(path, f) for f in os.scandir(path)
                ]

        return file_paths

    def get_images_by_artist(self):
        file_paths = {}

        for style_directory in os.scandir("../data/images"):
            sub_path = os.path.join("../data/images", style_directory)

            for artist_directory in os.scandir(sub_path):
                if artist_directory not in file_paths:
                    file_paths[artist_directory] = []

                path = os.path.join(sub_path, artist_directory)
                file_paths[artist_directory] += [
                    os.path.join(path, f) for f in os.scandir(path)
                ]

        return file_paths

    def initialize_pairings(self):
        # Get paths grouped by artist or artstyle
        # Then turn it into a 2D list
        grouped_paths = (
            self.get_images_by_artist()
            if self.pair_by == "artist"
            else self.get_images_by_style()
        )
        grouped_paths = [(label, list) for label, list in grouped_paths.items()]

        get_positive_pairs = (
            self.pairing_scheme == "positive" or self.pairing_scheme == "both"
        )
        get_negative_pairs = (
            self.pairing_scheme == "negative" or self.pairing_scheme == "both"
        )
        pairings = []

        # Positive pairs go over each list in the 2D list
        # And create all possible pairings between elements in that list
        if get_positive_pairs:
            for label, list in grouped_paths:
                for i, path_i in enumerate(list):
                    for path_j in list[i + 1 :]:
                        pairing_one = ((label, path_i), (label, path_j))
                        pairing_two = ((label, path_j), (label, path_i))
                        pairings.append(pairing_one)
                        pairings.append(pairing_two)

        # Negative pairs go over each pair of lists in the 2D list
        # And generates all possible pairs between those two lists
        if get_negative_pairs:
            for i, (label_i, list_i) in enumerate(grouped_paths):
                for label_j, list_j in grouped_paths[i + 1 :]:
                    for k, path_i in enumerate(list_i):
                        for path_j in list_j[k + 1 :]:
                            pairing_one = ((label_i, path_i), (label_j, path_j))
                            pairing_two = ((label_j, path_j), (label_i, path_i))
                            pairings.append(pairing_one)
                            pairings.append(pairing_two)

        return pairings
    
    def label_index_to_name(self, type, index):
        """
            Returns the name of a label, given that label's
            index and type

            type: 'artist' or 'style' -> 
                The type of label the index corresponds to

            index: int -> 
                The index of the label
        """
        if type == 'artist':
            return self.index_to_artist[index]
        
        return self.index_to_style[index]

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, idx):
        (label_1, path_1), (label_2, path_2) = self.pairings[idx]
        image_1 = read_image(path_1)
        image_2 = read_image(path_2)

        transformation = transforms.Compose(
            [
                torchvision.transforms.Resize((self.image_size, self.image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_1 = transformation(image_1)
        image_2 = torch.FloatTensor(image_2)

        if self.pair_by == "artist":
            label_1 = torch.tensor(self.artist_to_index[label_1])
            label_2 = torch.tensor(self.artist_to_index[label_2])
        else:
            label_1 = torch.tensor(self.style_to_index[label_1])
            label_2 = torch.tensor(self.style_to_index[label_2])

        return image_1, image_2, label_1, label_2

class ArtworkImageDatasetNoPairings(Dataset):
    def __init__(self, image_size):
        """
        image_size: int ->
            Specifies the (image_size, image_size) resolution of the image

        pair_by: 'artist' or 'style' ->
            Specifies how we should group up images in the dataset. For example,
            pair_by = 'artist' will have the dataset generate pairs (x, y) of images
            with the same artist. For pair_by = 'style', the dataset will generate pairs
            (x, y) of images with the same art style

        pairing_scheme: 'positive' or 'negative' or 'both' ->
            Specifies the pairing scheme of images (x, y) in the dataset.
            Positive means we will only have pairs (x, y) with the same grouping
            (E.g. (x, y) have the same artist / artstyle, depending on the value of pair_by)
            Negative means we will only have pairs (x, y) with DIFFERENT groupings
            Both means we will have both positive and negative pairs

        """
        self.image_size = image_size

        self.file_paths = []
        for style_directory in os.scandir("../data/images"):
            sub_path = os.path.join("../data/images", style_directory)

            for artist_directory in os.scandir(sub_path):
                path = os.path.join(sub_path, artist_directory)
                file_paths += [os.path.join(path, f) for f in os.scandir(path)]

        artists = [f.split("/")[-2] for f in self.file_paths]
        artists = list(set(artists))
        self.artist_to_index = {artist: artists.index(artist) for artist in artists}
        self.index_to_artist = {idx: artist for artist, idx in self.artist_to_index}

        styles = [f.split("/")[-3] for f in self.file_paths]
        styles = list(set(styles))
        self.style_to_index = {style: styles.index(style) for style in styles}
        self.index_to_style = {idx: style for style, idx in self.style_to_index}

    def __len__(self):
        return len(self.file_paths)

    def label_index_to_name(self, type, index):
        """
            Returns the name of a label, given that label's
            index and type

            type: 'artist' or 'style' -> 
                The type of label the index corresponds to

            index: int -> 
                The index of the label
        """
        if type == 'artist':
            return self.index_to_artist[index]
        
        return self.index_to_style[index]
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = read_image(path)

        transformation = transforms.Compose(
            [
                torchvision.transforms.Resize((self.image_size, self.image_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = transformation(image)

        artstyle = path.split('/')[-3]
        artstyle_idx = torch.tensor(self.style_to_index[artstyle])

        artist = path.split('/')[-2]
        artist_idx = torch.tensor(self.artist_to_index[artist])

        return image, artist_idx, artstyle_idx
