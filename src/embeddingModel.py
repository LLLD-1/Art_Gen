import torch
import torch.nn as nn


def is_power_2(n: int):
    return n.bit_count() == 1


class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dimension, image_size=256):
        """
        embedding_dimension: int ->
            The dimension of the final embedding vector output
            by the model
        """
        super(EmbeddingModel, self).__init__()

        assert is_power_2(image_size) and is_power_2(embedding_dimension)
        assert embedding_dimension < image_size
        # Model should keep cutting image size in half to get to embedding dimension
