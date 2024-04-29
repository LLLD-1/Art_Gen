import torch
import torch.nn as nn
from pix2pix import Generator, Pix2PixEmbeddor
from dataset import ArtworkImageDataset
import numpy as np
import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from plotter import plot_training_metrics


############################################################
############################################################
#! MODEL
############################################################
############################################################
class EvalModel(torch.nn.Module):
    def __init__(self, embedding_dimension, sizes=[50, 50, 50]):
        super(EvalModel, self).__init__()

        self.initial = nn.Linear(embedding_dimension * 2, sizes[0])

        layers = []
        input_dim = sizes[0]
        for size in sizes[1:]:
            layers.append(nn.Sequential(nn.Linear(input_dim, size), nn.LeakyReLU(0.2)))
            input_dim = size

        self.hidden = nn.Sequential(*layers)

        self.output = nn.Sequential(nn.Linear(sizes[-1], 1), nn.Sigmoid())

    def forward(self, x, y):
        x = torch.cat([x, y])
        x = self.initial(x)
        x = self.hidden(x)
        x = self.output(x)
        return x


############################################################
############################################################
#! HELPERS
############################################################
############################################################
def evaluate(model, embedding_model, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_fn = torch.nn.BCELoss()

    model = model.to(device)
    accuracy = []
    loss = []

    for x, y, labels_x, labels_y in loader:
        x = x.to(device)
        y = y.to(device)
        labels_x = labels_x.to(device)
        labels_y = labels_y.to(device)

        x_embed = embedding_model(x)
        y_embed = embedding_model(y)

        predictions = model(x_embed, y_embed)
        labels = labels_x.view(-1) == labels_y.view(-1)
        l = loss_fn(predictions, labels)
        loss.append(l)

        accuracy = predictions.argmax(1) == labels
        accuracy.append(accuracy)

    mean_loss = np.array(loss).mean()
    mean_accuracy = np.array(accuracy).mean()
    return mean_loss, mean_accuracy


def train(
    model,
    embedding_model,
    train_dataset,
    val_dataset,
    lr,
    epochs,
    batch_size,
    device,
    save_interval,
    model_name="model",
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    embedding_model = embedding_model.to(device)
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    progress_bar = tqdm(epochs)
    for epoch_idx in progress_bar:

        batch_loss = []
        for x, y, labels_x, labels_y in loader:
            print(labels_x, labels_y)
            x = x.to(device)
            y = y.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)

            x_embed = embedding_model(x)
            y_embed = embedding_model(y)

            predictions = model(x_embed, y_embed)
            labels = labels_x.view(-1) == labels_y.view(-1)
            loss = loss_fn(predictions, labels)

            accuracy = predictions.argmax(1) == labels
            train_accuracy.append(accuracy)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss.append(loss)

        mean_loss = np.array(batch_loss).mean()
        train_loss.append(mean_loss)

        val_l, val_a = evaluate(val_dataset, batch_size, device)
        val_loss.append(val_l)
        val_accuracy.append(val_a)

        if epoch_idx % 3 == 0:
            mean_loss = np.array(loss).mean()
            progress_bar.set_postfix(mean_loss)

        if save_interval is not None and epoch_idx % save_interval == 0:
            torch.save(
                model.state_dict(),
                f"../models/evalModel/{model_name}_epoch={epoch_idx}.tar",
            )

        return train_loss, train_accuracy, val_loss, val_accuracy


def get_embedding_model(type, path):
    if type == "pix2pix":
        generator = Generator()
        generator.load_state_dict(path)
        return Pix2PixEmbeddor(generator)

    # TODO: return the other model


############################################################
############################################################
#! DATASET
############################################################
############################################################

train_proportion = 0.70

dataset = ArtworkImageDataset(256, pair_by="style", pairing_scheme="both")

train_len = int(len(dataset) * train_proportion)
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

############################################################
############################################################
#! TRAINING
############################################################
############################################################

# Load the embeddings model

device = "cuda"
epochs = 10
lr = 0.001
batch_size = 16
save_interval = 3

model = EvalModel()
embedding_model = get_embedding_model("pix2pix", "../models/autoencoder/TODO")

train_loss, train_accuracy, val_loss, val_accuracy = train(
    model,
    embedding_model,
    train_dataset,
    val_dataset,
    lr,
    epochs,
    batch_size,
    device,
    save_interval,
)

plot_training_metrics(
    train_loss, val_loss, title="Eval Model Loss", y_label="Average Loss"
)
plot_training_metrics(
    train_accuracy,
    val_accuracy,
    title="Eval Model Accuracy",
    y_label="Average Accuracy",
)
