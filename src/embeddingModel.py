import torch.nn as nn


def is_power_2(n: int):
    return n.bit_count() == 1


class SqueezeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(SqueezeBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
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
        starting_features=16,
    ):
        """
        embedding_dimension: int ->
            The dimension of the final embedding vector output
            by the model
        """
        super(EmbeddingModel, self).__init__()

        assert is_power_2(image_size) and is_power_2(embedding_dimension)
        assert embedding_dimension < image_size

        self.initial = nn.Conv2d(
            in_channels,
            starting_features,
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect",
        )

        # Model should keep cutting image size in half to get to embedding dimension
        layers = []

        current_dim = image_size // 2
        current_features = starting_features
        while current_dim != embedding_dimension:
            layers.append(
                SqueezeBlock(current_features, current_features, dropout_prob)
            )
            current_dim //= 2
            current_features *= 2

        self.squeeze = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.squeeze(x)
        x = nn.Flatten(x)
        return x
