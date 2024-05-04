import torch.nn as nn
import torch


def is_power_2(n: int):
    return n.bit_count() == 1


class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(SqueezeBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        return self.block(x)


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        dropout_prob=0.2,
        in_channels=3,
        image_size=256,
    ):
        """
        embedding_dimension: int ->
            The dimension of the final embedding vector output
            by the model
        """
        super(EmbeddingModel, self).__init__()

        assert is_power_2(image_size)

        features = embedding_dimension

        self.initial = nn.Conv2d(
            in_channels,
            features,
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect",
        )

        # Model should keep cutting image size in half to get to embedding dimension
        layers = []

        current_dim = image_size // 2
        while current_dim != 0:
            layers.append(
                SqueezeBlock(features, features, dropout_prob)
            )
            current_dim //= 2

        self.squeeze = nn.Sequential(*layers)
        self.final = nn.Conv2d(
            features,
            features,
            kernel_size=2,
            stride=1,
            padding=0,
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.initial(x)
        x = self.squeeze(x)
        x = self.final(x)
        x = self.flatten(x)
        return x


def test():
    model = EmbeddingModel(embedding_dimension=8)
    x = torch.ones((1, 3, 256, 256))
    e = model(x)
    print(e.shape)


if __name__ == '__main__':
    test()
