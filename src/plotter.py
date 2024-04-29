import matplotlib.pyplot as plt
import numpy as np

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

def plot_pca(dataset, embedding_model):
    # Go over the dataset, PCA each embedding vector to dim=2
    # Then plot the points on a graph with the colors representing the art style
    pass