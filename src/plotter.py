import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from src.dataset import ArtworkImageDatasetNoPairings
from tqdm import tqdm


def plot_training_metrics(train_data, validation_data, title, y_label):
    assert len(train_data) == len(validation_data)
    epochs = np.arange(len(train_data))

    # plot the losses
    plt.plot(epochs, train_data, label="Train")
    plt.plot(epochs, validation_data, label="Validation")
    plt.xlabel("# Epochs")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()


def plot_pca(embedding_model, group_by="style", device="cuda", batch_size=16):
    """
    Plots the embedded representation of the dataset's images in 2 dimensions
    using the embedding model and PCA. Datapoints on the graph are color coded
    according to the 'group by' parameter.

    group_by: 'artist' or 'style' ->
        Whether to classify (color) points based on artist or art style
    """
    embedding_model = embedding_model.to(device)

    dataset = ArtworkImageDatasetNoPairings(256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    embedding_vectors = {}
    for x, artist_labels, artstyle_labels in tqdm(loader):
        x = x.to(device)
        x_embedd = embedding_model(x)

        labels_to_group_by = artstyle_labels if group_by == "style" else artist_labels
        labels_to_group_by = labels_to_group_by.view(-1)

        for i, label in enumerate(labels_to_group_by):
            if label not in embedding_vectors:
                embedding_vectors[label] = []

            vec = x_embedd[i].cpu().numpy()
            embedding_vectors[label].append(vec)

    # Concatenate all vectors lists into one big list
    all_vectors = sum(list(embedding_vectors.values()), [])
    all_vectors = np.array(all_vectors)

    std_scaler = StandardScaler()
    pca = PCA(n_components=2)

    normalized = std_scaler.fit_transform(list)
    projected = pca.fit_transform(normalized)

    # We want each label group to be its own color
    running_size = 0
    for label_idx, list in embedding_vectors.items():
        size = len(list)
        slice = projected[running_size : running_size + size]
        label_name = dataset.label_index_to_name(group_by, label_idx)

        plt.scatter(slice[:, 0], slice[:, 1], label=label_name)

        running_size += size

    plt.legend()
    plt.show()
