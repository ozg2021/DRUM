import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 128):
        """
        Initializes the TimeEmbedding layer.

        Args:
            emb_dim (int): The dimensionality of the time embedding. It must be divisible by 8. Default is 128.
        """
        super(TimeEmbedding, self).__init__()

        assert emb_dim % 8 == 0, "emb_dim must be divisible by 8"

        self.emb_dim = emb_dim
        self.lin1 = nn.Linear(self.emb_dim // 4, self.emb_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(self.emb_dim, self.emb_dim)


    def forward(self, t: torch.Tensor):
        """
        Forward pass for the time embedding.

        Args:
            t (torch.Tensor): A tensor containing time values, shape (batch_size,)

        Returns:
            torch.Tensor: The time embedding, shape (batch_size, emb_dim)
        """
        dim = self.emb_dim // 8

        emb = math.log(10000) / (dim - 1)
        emb = torch.exp(torch.arange(dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb
