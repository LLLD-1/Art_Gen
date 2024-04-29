import torch
from pix2pix import Generator, Discriminator
from dataset import ArtworkImageDataset
import numpy as np
import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from plotter import plot_training_metrics


############################################################
############################################################
#! HELPERS
#! Training loop adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/train.py
############################################################
############################################################
def evaluate(generator, discriminator, dataset, batch_size, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss = []
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for x, _, _, _ in loader:
        x = x.to(device)

        # Generator Loss
        with torch.cuda.amp.autocast():
            _, fake_images = generator(x)
            D_fake = discriminator(x, fake_images)
            G_fake_loss = loss(D_fake, torch.ones_like(D_fake))
            G_loss = G_fake_loss

        loss.append(G_loss)

    mean_loss = np.array(loss).mean()
    return mean_loss


def train(
    generator,
    discriminator,
    train_dataset,
    val_dataset,
    lr,
    epochs,
    batch_size,
    device,
    save_interval,
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    loss = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    train_loss = []
    val_loss = []

    progress_bar = tqdm(epochs)
    for epoch_idx in progress_bar:

        batch_loss = []
        for x, y, labels_x, labels_y in loader:
            print(labels_x, labels_y)
            x = x.to(device)
            y = y.to(device)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                _, fake_images = generator(x)

                D_real = discriminator(x, y)
                D_real_loss = loss(D_real, torch.ones_like(D_real))

                D_fake = discriminator(x, fake_images.detach())
                D_fake_loss = loss(D_fake, torch.zeros_like(D_fake))

                D_loss = (D_real_loss + D_fake_loss) / 2

            D_loss.backward()
            optimizer_d.step()
            optimizer_d.zero_grad()

            # Train Generator
            with torch.cuda.amp.autocast():
                D_fake = discriminator(x, fake_images)
                G_fake_loss = loss(D_fake, torch.ones_like(D_fake))
                G_loss = G_fake_loss

            G_loss.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()
            batch_loss.append(G_loss)

        mean_loss = np.array(batch_loss).mean()
        train_loss.append(mean_loss)

        val_l = evaluate(val_dataset, batch_size, device)
        val_loss.append(val_l)

        if epoch_idx % 3 == 0:
            mean_loss = np.array(loss).mean()
            progress_bar.set_postfix(mean_loss)

        if save_interval is not None and epoch_idx % save_interval == 0:
            torch.save(
                generator.state_dict(),
                f"../models/autoencoder/generator_epoch={epoch_idx}.tar",
            )
            torch.save(
                discriminator.state_dict(),
                f"../models/autoencoder/discriminator_epoch={epoch_idx}.tar",
            )

        return train_loss, val_loss

############################################################
############################################################
#! DATASET
############################################################
############################################################

train_proportion = 0.70

dataset = ArtworkImageDataset(256, pair_by='artist', pairing_scheme='positive')

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
save_interval = 3

generator = Generator()
discriminator = Discriminator()

train_loss, val_loss = train(
    generator,
    discriminator,
    train_dataset,
    val_dataset,
    lr,
    epochs,
    batch_size,
    device,
    save_interval,
)

plot_training_metrics(train_loss, val_loss, title='Loss', y_label='Average Loss')
