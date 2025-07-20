import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, emb_dim=128, xs_dim=27):
        """
        Initializes the MLP model for processing the input and combined embeddings (unconditional diffusion model).

        Args:
            input_size (int): The size of the input feature vector.
            hidden_size (int): The number of hidden units in each MLP layer.
            emb_dim (int): The dimensionality of the time embedding. Default is 128.
            xs_dim (int): The dimensionality of the static attribute embedding. Default is 27.
        """
        super(MLP, self).__init__()

        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.xs_dim = xs_dim

        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size + emb_dim + xs_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size + emb_dim + xs_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size + emb_dim + xs_dim, hidden_size)
        ])

    def forward(self, x, combined_emb):
        """
        Forward pass for the MLP model.

        Args:
            x (torch.Tensor): The input feature tensor, shape (batch_size, input_size).
            combined_emb (torch.Tensor): The combined embedding tensor, shape (batch_size, emb_dim + xs_dim + hidden_size).

        Returns:
            torch.Tensor: The output of the MLP, shape (batch_size, hidden_size).
        """
        for i in range(0, len(self.layers) - 1, 2):
            x = self.layers[i](x)
            x = self.layers[i + 1](x)
            x = torch.cat((x, combined_emb), dim=1)

        out = self.layers[-1](x)
        return out
