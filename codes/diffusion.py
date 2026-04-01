import torch
import torch.nn as nn
from typing import Optional, Tuple


class Diffusion(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        Initializes the Diffusion process.

        Args:
            eps_model (nn.Module): The epsilon model (encoder-decoder LSTM).
            n_steps (int): The number of steps in the diffusion process.
            device (torch.device): The device to run computations on (e.g., "cuda" or "cpu").
        """
        super(Diffusion, self).__init__()
        self.eps_model = eps_model
        self.device = device
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta


    def _gather(self, consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gather constants for the time steps and reshape to the feature map shape.

        Args:
            consts (torch.Tensor): The constant tensor (e.g., alpha_bar values).
            t (torch.Tensor): The tensor representing time step indices.

        Returns:
            torch.Tensor: The gathered constants reshaped to the feature map shape.
        """
        c = consts.gather(-1, t)
        return c.reshape(-1, 1)


    def _gather_sample(self, consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gather constants for time step with device compatibility.

        Args:
            consts (torch.Tensor): The constant tensor (e.g., alpha_bar values).
            t (torch.Tensor): The tensor representing time step (i.e., noise level step) indices.

        Returns:
            torch.Tensor: The gathered constants reshaped to the feature map shape.
        """
        consts = consts.to(self.device)
        t = t.to(self.device)
        c = consts.gather(-1, t)
        return c.reshape(-1, 1)


    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the distribution q(x_t | x_0) in the forward process.

        Args:
            x0 (torch.Tensor): The original data (before diffusion).
            t (torch.Tensor): The time step indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and variance of the distribution.
        """
        mean = self._gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - self._gather(self.alpha_bar, t)
        return mean, var


    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: Sample from q(x_t | x_0).

        Args:
            x0 (torch.Tensor): The original data.
            t (torch.Tensor): The time step indices.
            eps (Optional[torch.Tensor]): The noise tensor (default is random noise).

        Returns:
            torch.Tensor: The noisy sample at time step 't'.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps


    def loss_conditional(self, x0, c, xs, q_std, noise=None):
        """
        Compute the conditional loss for the model based on the noisy samples.
        Uses inverse-variance weighting by basin discharge std.
 
        Args:
            x0 (torch.Tensor): The original data (before diffusion).
            c (torch.Tensor): Conditional inputs for the model (e.g., class labels).
            xs (torch.Tensor): The current state of the data (after some diffusion steps).
            q_std (float): Standard deviation of discharge for the current basin,
                          used for inverse-variance weighting.
            noise (Optional[torch.Tensor]): The noise added to the data (default is None).
 
        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long).to(self.device)
 
        if noise is None:
            noise = torch.randn_like(x0)
 
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t / self.n_steps, c, xs)
 
        weight = 1.0 / (q_std + 1e-8) ** 2
        loss = torch.mean(weight * (noise - eps_theta) ** 2)
        return loss
 
 
    def loss_unconditional(self, x0, xs, q_std, noise=None):
        """
        Compute the unconditional loss for the model based on noisy samples.
        Uses inverse-variance weighting by basin discharge std.
 
        Args:
            x0 (torch.Tensor): The original data (before diffusion).
            xs (torch.Tensor): The current state of the data (after some diffusion steps).
            q_std (float): Standard deviation of discharge for the current basin,
                          used for inverse-variance weighting.
            noise (Optional[torch.Tensor]): The noise added to the data (default is None).
 
        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long).to(self.device)
 
        if noise is None:
            noise = torch.randn_like(x0)
 
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t / self.n_steps, xs)
 
        weight = 1.0 / (q_std + 1e-8) ** 2
        loss = torch.mean(weight * (noise - eps_theta) ** 2)
        return loss

    # def loss_conditional(self, x0, c, xs, noise=None):
    #     """
    #     Compute the conditional loss for the model based on the noisy samples.

    #     Args:
    #         x0 (torch.Tensor): The original data (before diffusion).
    #         c (torch.Tensor): Conditional inputs for the model (e.g., class labels).
    #         xs (torch.Tensor): The current state of the data (after some diffusion steps).
    #         noise (Optional[torch.Tensor]): The noise added to the data (default is None).

    #     Returns:
    #         torch.Tensor: The computed loss.
    #     """
    #     batch_size = x0.shape[0]
    #     t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long).to(self.device)

    #     if noise is None:
    #         noise = torch.randn_like(x0)

    #     xt = self.q_sample(x0, t, eps=noise)
    #     eps_theta = self.eps_model(xt, t / self.n_steps, c, xs)

    #     loss = torch.norm(noise - eps_theta, p=2, dim=1).pow(2).sum()
    #     # loss = torch.mean((noise - eps_theta) ** 2)
    #     return loss


    # def loss_unconditional(self, x0, xs, noise=None):
    #     """
    #     Compute the unconditional loss for the model based on noisy samples.

    #     Args:
    #         x0 (torch.Tensor): The original data (before diffusion).
    #         xs (torch.Tensor): The current state of the data (after some diffusion steps).
    #         noise (Optional[torch.Tensor]): The noise added to the data (default is None).

    #     Returns:
    #         torch.Tensor: The computed loss.
    #     """
    #     batch_size = x0.shape[0]
    #     t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long).to(self.device)

    #     if noise is None:
    #         noise = torch.randn_like(x0)

    #     xt = self.q_sample(x0, t, eps=noise)
    #     eps_theta = self.eps_model(xt, t / self.n_steps, xs)

    #     loss = torch.norm(noise - eps_theta, p=2, dim=1).pow(2).sum()
    #     # loss = torch.mean((noise - eps_theta) ** 2)
    #     return loss

