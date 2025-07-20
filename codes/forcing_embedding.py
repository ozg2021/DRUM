import torch
import torch.nn as nn


class ForcingEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Initializes the ForcingEmbedding layer, which uses an LSTM for generating embeddings.

        Args:
            input_size (int): The number of input features for each time step (e.g., the size of the input data at each time step).
            hidden_size (int): The number of units in the LSTM's hidden layer.
        """
        super(ForcingEmbedding, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self._init_weights()


    def _init_weights(self):
        """
        Initialize LSTM weights with Xavier uniform distribution.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'bias_ih' in name:
                    param.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)


    def forward(self, xd):
        """
        Forward pass for the ForcingEmbedding layer. Processes the input sequence using LSTM.

        Args:
            xd (torch.Tensor): Input tensor containing sequential data, shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The last hidden state of the LSTM, shape (batch_size, hidden_size).
        """
        batch_size, seq_len, _ = xd.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=xd.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=xd.device)

        lstm_out, (h_n, c_n) = self.lstm(xd, (h0, c0))

        return h_n[-1]
