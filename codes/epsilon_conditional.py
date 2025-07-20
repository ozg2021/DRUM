import torch
import torch.nn as nn
from codes.time_embedding import TimeEmbedding
from codes.forcing_embedding import ForcingEmbedding
from codes.mlp_conditional import MLP


class EpsModel(nn.Module):
    """
    EpsModel: A conditional diffusion model for noise prediction.

    This model combines temporal, static, and forcing information to predict noise
    in the diffusion process. It uses LSTM-based encoders and MLPs for feature
    extraction and integration.
    """

    def __init__(self, input_size=7, static_attr_len=27, hidden_size=256, emb_dim=128, future_step=1, num_layers=1):
        """
        Initialize the EpsModel, a diffusion conditional model for noise prediction.

        Args:
            input_size (int): The size of the input feature vector (e.g., meteorological variables).
            static_attr_len (int): The length of the static attributes (e.g., basin characteristics).
            hidden_size (int): The size of the hidden layer in LSTM and MLP components.
            emb_dim (int): The dimensionality of the time embedding vector.
            future_step (int): The number of future time steps to predict.
            num_layers (int): Number of LSTM layers for sequential processing.
        """
        super(EpsModel, self).__init__()

        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.future_step = future_step
        self.num_layers = num_layers

        self.time_embedding = TimeEmbedding(emb_dim)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_attr_len, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )

        self.forcing_embedding = ForcingEmbedding(input_size + 128, hidden_size, num_layers)

        self.mlp = MLP(1 + hidden_size + emb_dim + 128, hidden_size, emb_dim=128, xs_dim=128)

        self.decoder_lstm = nn.LSTM(
            input_size=1 + hidden_size + emb_dim + 128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 8),
            nn.GELU(),
            nn.Linear(hidden_size // 8, self.future_step)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for the decoder LSTM using best practices.

        - Input-to-hidden weights: Xavier uniform initialization
        - Hidden-to-hidden weights: Orthogonal initialization
        - Biases: Zero initialization with forget gate bias set to 1.0
        """
        for name, param in self.decoder_lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'bias_ih' in name:
                    param.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, xt, t, xd, xs):
        """
        Forward pass through the diffusion noise prediction model.

        Args:
            xt (torch.Tensor): Current noisy state at diffusion time step t.
                              Shape: (batch_size, 1). Represents the noisy target variable.
            t (torch.Tensor): Diffusion time step values for time embedding.
                             Shape: (batch_size,). Values typically in [0, n_steps-1].
            xd (torch.Tensor): Input sequence data (meteorological forcing).
                              Shape: (batch_size, seq_len, input_size).
                              Contains historical meteorological observations.
            xs (torch.Tensor): Static basin attributes (topography, soil, etc.).
                              Shape: (batch_size, static_attr_len).
                              Time-invariant characteristics of the watershed.

        Returns:
            torch.Tensor: Predicted noise at the current diffusion step.
                         Shape: (batch_size, future_step).
                         Used to denoise xt in the reverse diffusion process.
        """
        batch_size, seq_len, _ = xd.size()

        t_emb = self.time_embedding(t)
        xs_emb = self.static_mlp(xs)
        xs_emb_expanded = xs_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        xd_combined = torch.cat((xd, xs_emb_expanded), dim=2)
        xd_emb = self.forcing_embedding(xd_combined)
        combined_emb = torch.cat((t_emb, xd_emb, xs_emb), dim=1)
        combined_input = torch.cat((xt, t_emb, xd_emb, xs_emb), dim=1)
        decoder_input = combined_input.unsqueeze(1)

        H = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(xt.device)
        C = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(xt.device)

        out, _ = self.decoder_lstm(decoder_input, (H, C))
        lstm_output = out.squeeze(1)
        mlp_output = self.mlp(combined_input, combined_emb)
        combined_out = torch.cat((lstm_output, mlp_output), dim=1)
        out = self.fc(combined_out)

        return out
