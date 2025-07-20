import torch
from torch.cuda.amp import autocast


def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Select the corresponding value from the constant tensor based on the time step t.

    Args:
        consts (torch.Tensor): Constant tensor.
        t (torch.Tensor): Time step tensor.

    Returns:
        torch.Tensor: The selected value with shape [batch_size, 1].
    """
    if consts.device != t.device:
        consts = consts.to(t.device)
    return consts.gather(-1, t).reshape(-1, 1)


def get_timestep_embedding(n_steps: int, sampling_steps: int, time_steps: torch.Tensor) -> torch.Tensor:
    """
    Linearly map the sampling steps to the training steps for DDIM acceleration.

    Args:
        n_steps (int): Total number of training steps (e.g., 1000).
        sampling_steps (int): Number of sampling steps (e.g., 50).
        time_steps (torch.Tensor): Tensor representing the sampling steps.

    Returns:
        torch.Tensor: The mapped time steps tensor corresponding to training steps.
    """
    stride = n_steps // sampling_steps
    return (time_steps * stride).long()


def DDIM(xt, t, c, xs, eps_conditional, eps_unconditional, n_steps, sampling_steps, guide_w=0.2, eta=0.0):
    """
    Perform sampling using DDIM (Denoising Diffusion Implicit Models).

    Args:
        xt (torch.Tensor): Current state x_t.
        t (torch.Tensor): Time step tensor with shape [batch_size].
        c (torch.Tensor): Conditional data (e.g., precipitation data).
        xs (torch.Tensor): Static properties (e.g., basin characteristics).
        eps_conditional (function): Conditional noise prediction model.
        eps_unconditional (function): Unconditional noise prediction model.
        n_steps (int): Total number of training steps.
        sampling_steps (int): Number of sampling steps.
        guide_w (float, optional): Guidance weight for classifier-free guidance. Default is 0.2.
        eta (float, optional): Parameter controlling randomness (0=deterministic, >0=stochastic). Default is 0.0.

    Returns:
        torch.Tensor: The updated x_prev (previous state).
    """
    with torch.no_grad(), autocast():
        beta = torch.linspace(0.0001, 0.02, n_steps).to(xs.device)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        t_scaled = get_timestep_embedding(n_steps, sampling_steps, t)

        eps1 = eps_conditional(xt, t_scaled / float(n_steps), c, xs)
        eps2 = eps_unconditional(xt, t_scaled / float(n_steps), xs)

        eps = (1 + guide_w) * eps1 - guide_w * eps2

        alpha_bar_current = gather(alpha_bar, t_scaled)

        prev_t = (t - 1).clamp(min=0)
        prev_t_scaled = get_timestep_embedding(n_steps, sampling_steps, prev_t)
        alpha_bar_prev = gather(alpha_bar, prev_t_scaled)

        alpha_bar_prev = torch.where(t.view(-1, 1) > 0,
                                     alpha_bar_prev,
                                     torch.ones_like(alpha_bar_current))

        epsilon = 1e-8
        pred_x0 = (xt - torch.sqrt(1 - alpha_bar_current) * eps) / (torch.sqrt(alpha_bar_current) + epsilon)

        c1 = torch.sqrt(alpha_bar_prev)
        c2 = torch.sqrt(1 - alpha_bar_prev)
        x_prev = c1 * pred_x0 + c2 * eps

        if eta > 0:
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_current) * (1 - alpha_bar_current / alpha_bar_prev))
            noise = torch.randn_like(xt)
            x_prev = x_prev + sigma * noise

    return x_prev
