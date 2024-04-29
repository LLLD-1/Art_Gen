import torch
import torch
from dataset import ArtworkImageDataset
import numpy as np
import tqdm
from torch.utils.data import DataLoader, random_split
from plotter import plot_training_metrics, plot_pca
from embeddingModel import EmbeddingModel

############################################################
############################################################
#! HELPERS
############################################################
############################################################


def dot_product_loss(x, y, mode):
    """
    x : (batch_size, embedding_dimension)
    y : (batch_size, embedding_dimension)
    mode : 'maximize' or 'minimize'

    This loss with either maximize or minimize the dot product
    between every embedding vector in the rows of x and y.

    We calculate all dot products between x and why by computing
    x.T @ y. This gives us back a matrix

    We then sum all those dot products up using torch.sum
    """
    x_t = torch.transpose(x, 0, 1)
    norm = torch.sum(torch.matmul(x_t, y))

    if mode == "maximize":
        return -norm
    return norm


def evaluate(
    model,
    positive_weight,
    negative_weight,
    regularizer_weight,
    dataset,
    batch_size,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss_history = []
    model = model.to(device)

    for x, y, labels_x, labels_y in loader:
        x = x.to(device)
        y = y.to(device)
        labels_x = labels_x.to(device)
        labels_y = labels_y.to(device)

        # Group inputs into positive and negative pairs
        positive_pairs_indices = labels_x == labels_y
        x_positive = torch.index_select(x, 0, positive_pairs_indices)
        y_positive = torch.index_select(y, 0, positive_pairs_indices)

        negative_pairs_indices = labels_x != labels_y
        x_negative = torch.index_select(x, 0, negative_pairs_indices)
        y_negative = torch.index_select(y, 0, negative_pairs_indices)

        # Create the embeddings
        x_positive_embedd = model(x_positive)
        y_positive_embedd = model(y_positive)

        x_negative_embed = model(x_negative)
        y_negative_embed = model(y_negative)

        # Now compute the loss
        postive_loss = dot_product_loss(
            x_positive_embedd, y_positive_embedd, "maximize"
        )
        postive_loss *= positive_weight

        negative_loss = dot_product_loss(x_negative_embed, y_negative_embed, "minimize")
        negative_loss *= negative_weight
        # TODO: Also try to backpropgate each loss separetely instead of averaging them
        loss = (postive_loss + negative_loss) / 2

        # Add regularization
        params = torch.cat([p.view(-1) for p in model.parameters()])
        norm = torch.norm(params, 2)
        loss += regularizer_weight * norm
        loss_history.append(loss)

    mean_loss = np.array(loss).mean()
    return mean_loss


def train(
    model,
    train_dataset,
    val_dataset,
    lr,
    positive_weight,
    negative_weight,
    regularizer_weight,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    train_loss = []
    val_loss = []

    progress_bar = tqdm(epochs)
    for epoch_idx in progress_bar:

        batch_loss = []
        for x, y, labels_x, labels_y in loader:
            x = x.to(device)
            y = y.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)

            # Group inputs into positive and negative pairs
            positive_pairs_indices = labels_x == labels_y
            x_positive = torch.index_select(x, 0, positive_pairs_indices)
            y_positive = torch.index_select(y, 0, positive_pairs_indices)

            negative_pairs_indices = labels_x != labels_y
            x_negative = torch.index_select(x, 0, negative_pairs_indices)
            y_negative = torch.index_select(y, 0, negative_pairs_indices)

            # Create the embeddings
            x_positive_embedd = model(x_positive)
            y_positive_embedd = model(y_positive)

            x_negative_embed = model(x_negative)
            y_negative_embed = model(y_negative)

            # Now compute the loss
            postive_loss = dot_product_loss(
                x_positive_embedd, y_positive_embedd, "maximize"
            )
            postive_loss *= positive_weight

            negative_loss = dot_product_loss(
                x_negative_embed, y_negative_embed, "minimize"
            )
            negative_loss *= negative_weight
            # TODO: Also try to backpropgate each loss separetely instead of averaging them
            loss = (postive_loss + negative_loss) / 2

            # Add regularization
            params = torch.cat([p.view(-1) for p in model.parameters()])
            norm = torch.norm(params, 2)
            loss += regularizer_weight * norm

            # And backpropagate it
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss.append(loss)

        mean_loss = np.array(batch_loss).mean()
        train_loss.append(mean_loss)

        val_l = evaluate(
            val_dataset,
            positive_weight,
            negative_weight,
            regularizer_weight,
            batch_size,
            device,
        )
        val_loss.append(val_l)

        if epoch_idx % 3 == 0:
            mean_loss = np.array(loss).mean()
            progress_bar.set_postfix(mean_loss)

        if save_interval is not None and epoch_idx % save_interval == 0:
            torch.save(
                model.state_dict(),
                f"../models/embeddor/{model_name}_epoch={epoch_idx}.tar",
            )

        return train_loss, val_loss


############################################################
############################################################
#! DATASET
############################################################
############################################################

train_proportion = 0.70

# TODO: Try 'positive' and 'both' pairing schemes
dataset = ArtworkImageDataset(256, pair_by="artist", pairing_scheme="both")

train_len = int(len(dataset) * train_proportion)
val_len = len(dataset) - train_len

train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

############################################################
############################################################
#! TRAINING
############################################################
############################################################
device = "cuda"
epochs = 10
lr = 0.001
batch_size = 16
positive_weight = 1
negative_weight = 0.5
regularizer_weight = 0.1
save_interval = 3

model = EmbeddingModel(16, starting_features=16)

train_loss, train_accuracy, val_loss, val_accuracy = train(
    model,
    train_dataset,
    val_dataset,
    lr,
    positive_weight,
    negative_weight,
    regularizer_weight,
    epochs,
    batch_size,
    device,
    save_interval,
)

plot_training_metrics(
    train_loss, val_loss, title="Embedding Model Loss", y_label="Average Loss"
)
plot_pca(model, group_by='artist')
