import torch
import torch.nn as nn
from codes.time_embedding import TimeEmbedding
from codes.mlp_unconditional import MLP


class EpsModel(nn.Module):
    """
    EpsModel: An unconditional diffusion model for noise prediction.

    This model predicts noise in the diffusion process using only static basin
    characteristics and time information, without meteorological forcing data.
    It is used for classifier-free guidance in conditional diffusion models.
    """

    def __init__(self, static_attr_len=27, hidden_size=256, emb_dim=128, future_step=1, num_layers=1):
        """
        Initialize the unconditional EpsModel for noise prediction.

        Args:
            static_attr_len (int): Length of the static attributes vector (e.g., basin characteristics).
            hidden_size (int): Size of the hidden layers in LSTM and MLP components.
            emb_dim (int): The dimensionality of the time embedding vector.
            future_step (int): Number of future time steps to predict.
            num_layers (int): Number of LSTM layers for sequential processing.
        """
        super(EpsModel, self).__init__()

        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.future_step = future_step
        self.static_attr_len = static_attr_len
        self.num_layers = num_layers

        self.time_embedding = TimeEmbedding(emb_dim)

        self.static_mlp = nn.Sequential(
            nn.Linear(static_attr_len, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )

        self.mlp = MLP(input_size=1 + emb_dim + 128, hidden_size=hidden_size,
                       emb_dim=emb_dim, xs_dim=128)

        self.decoder_lstm = nn.LSTM(
            input_size=1 + emb_dim + 128,
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

    def forward(self, xt, t, xs):
        """
        Forward pass through the unconditional diffusion noise prediction model.

        Args:
            xt (torch.Tensor): Current noisy state at diffusion time step t.
                              Shape: (batch_size, 1). Represents the noisy target variable.
            t (torch.Tensor): Diffusion time step values for time embedding.
                             Shape: (batch_size,). Values typically in [0, n_steps-1].
            xs (torch.Tensor): Static basin attributes (topography, soil, etc.).
                              Shape: (batch_size, static_attr_len).
                              Time-invariant characteristics of the watershed.

        Returns:
            torch.Tensor: Predicted noise at the current diffusion step.
                         Shape: (batch_size, future_step).
                         Used for unconditional guidance in the reverse diffusion process.
        """
        batch_size = xt.size(0)

        assert xs.dim() == 2, f"Static attributes xs must be 2D tensor, current dim: {xs.dim()}"
        assert xs.size(1) == self.static_attr_len, f"Static attribute dimension mismatch, expected: {self.static_attr_len}, actual: {xs.size(1)}"

        t_emb = self.time_embedding(t)
        xs_emb = self.static_mlp(xs)
        combined_emb = torch.cat((t_emb, xs_emb), dim=1)
        combined_input = torch.cat((xt, combined_emb), dim=1)
        decoder_input = combined_input.unsqueeze(1)

        H = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(xt.device)
        C = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(xt.device)

        out, _ = self.decoder_lstm(decoder_input, (H, C))
        lstm_output = out.squeeze(1)
        mlp_output = self.mlp(combined_input, combined_emb)
        combined_out = torch.cat((lstm_output, mlp_output), dim=1)
        out = self.fc(combined_out)

        return out


class EpsModel_nowcast(nn.Module):
    """
    EpsModel_nowcast: An unconditional diffusion model for single-step nowcasting.
 
    Simplified version of EpsModel that removes the decoder LSTM branch,
    using only the MLP path followed by a lightweight FC head for noise prediction.
    """
 
    def __init__(self, static_attr_len=27, hidden_size=256, emb_dim=128, future_step=1):
        """
        Initialize the unconditional EpsModel_nowcast for noise prediction.
 
        Args:
            static_attr_len (int): Length of the static attributes vector (e.g., basin characteristics).
            hidden_size (int): Size of the hidden layers in MLP components.
            emb_dim (int): The dimensionality of the time embedding vector.
            future_step (int): Number of future time steps to predict.
        """
        super(EpsModel_nowcast, self).__init__()
 
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.future_step = future_step
        self.static_attr_len = static_attr_len
 
        self.time_embedding = TimeEmbedding(emb_dim)
 
        self.static_mlp = nn.Sequential(
            nn.Linear(static_attr_len, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )
 
        self.mlp = MLP(input_size=1 + emb_dim + 128, hidden_size=hidden_size,
                       emb_dim=emb_dim, xs_dim=128)
 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, future_step)
        )
 
    def forward(self, xt, t, xs):
        """
        Forward pass through the unconditional nowcast noise prediction model.
 
        Args:
            xt (torch.Tensor): Current noisy state. Shape: (batch_size, 1).
            t (torch.Tensor): Diffusion time step. Shape: (batch_size,).
            xs (torch.Tensor): Static basin attributes.
                              Shape: (batch_size, static_attr_len).
 
        Returns:
            torch.Tensor: Predicted noise. Shape: (batch_size, future_step).
        """
        assert xs.dim() == 2, f"Static attributes xs must be 2D tensor, current dim: {xs.dim()}"
        assert xs.size(1) == self.static_attr_len, f"Static attribute dimension mismatch, expected: {self.static_attr_len}, actual: {xs.size(1)}"
 
        t_emb = self.time_embedding(t)
        xs_emb = self.static_mlp(xs)
        combined_emb = torch.cat((t_emb, xs_emb), dim=1)
        combined_input = torch.cat((xt, combined_emb), dim=1)
 
        mlp_output = self.mlp(combined_input, combined_emb)
        out = self.fc(mlp_output)
 
        return out
 