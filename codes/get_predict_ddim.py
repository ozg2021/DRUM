import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd
from codes.DDIM import DDIM


def calculate_nse(predicted, observed):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    """
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - numerator / denominator
    return nse


def save_data(basin_id, sample, true, sample_mean, directory):
    """
    Save the prediction results and true values to files.

    Args:
        basin_id (str): Basin identifier.
        sample (np.ndarray): Multiple sample predictions with shape [time_steps, num_samples].
        true (np.ndarray): True observed values with shape [time_steps].
        sample_mean (np.ndarray): Mean predictions with shape [time_steps].
        directory (str): Base directory for saving files.
    """
    basin_dir = os.path.join(directory, basin_id)
    if not os.path.exists(basin_dir):
        os.makedirs(basin_dir)

    np.save(os.path.join(basin_dir, 'sample.npy'), sample)
    np.save(os.path.join(basin_dir, 'true.npy'), true)
    np.save(os.path.join(basin_dir, 'sample_mean.npy'), sample_mean)


def sample_ddim(c, xs, eps_conditional, eps_unconditional, n_steps, sampling_steps, clip_min=0, eta=0.0, guide_w=0.2):
    """
    Perform sampling using DDIM algorithm.

    Args:
        c (torch.Tensor): Conditional data (e.g., meteorological inputs) with shape [batch_size, seq_len, features].
        xs (torch.Tensor): Static properties (e.g., basin characteristics) with shape [batch_size, static_features].
        eps_conditional (nn.Module): Conditional noise prediction model.
        eps_unconditional (nn.Module): Unconditional noise prediction model.
        n_steps (int): Total number of training diffusion steps.
        sampling_steps (int): Number of sampling steps (can be much smaller than n_steps).
        clip_min (float, optional): Minimum value for clipping output. Default is 0.
        eta (float, optional): Stochasticity parameter (0=deterministic, >0=stochastic). Default is 0.0.
        guide_w (float, optional): Classifier-free guidance weight. Default is 0.2.

    Returns:
        torch.Tensor: Generated samples with shape [batch_size, output_dim].
    """
    eps_conditional.eval()
    eps_unconditional.eval()

    with torch.no_grad(), autocast():
        sample_num = c.shape[0]
        x = torch.randn(sample_num, 1, device=xs.device)

        time_steps = torch.arange(sampling_steps, device=xs.device).flip(0)

        for t in time_steps:
            time_tensor = torch.full((sample_num,), t, dtype=torch.long, device=xs.device)

            x = DDIM(x, time_tensor, c, xs, eps_conditional, eps_unconditional,
                     n_steps, sampling_steps, guide_w=guide_w, eta=eta)

        x = torch.clamp(x, min=clip_min)

    return x


def get_predict(eps_conditional, eps_unconditional, n_steps, sampling_steps, basin_list, selected_indices,
                testloader_list, normal_attribute, load_attribute, batch_size, num_samples, eta,
                guide_w, clip_min, output_base_dir, device):
    """
    Generate predictions for multiple basins and save results with NSE evaluation.

    Args:
        eps_conditional (nn.Module): Conditional noise prediction model.
        eps_unconditional (nn.Module): Unconditional noise prediction model.
        n_steps (int): Total number of training diffusion steps.
        sampling_steps (int): Number of sampling steps for DDIM.
        basin_list (list): List of all basin identifiers.
        selected_indices (list): Indices of basins to process.
        testloader_list (list): List of test data loaders for each basin.
        normal_attribute (object): Normalization parameters for static attributes.
        load_attribute (function): Function to load basin static attributes.
        batch_size (int): Batch size for processing.
        num_samples (int): Number of samples to generate for uncertainty quantification.
        eta (float): Stochasticity parameter for DDIM.
        guide_w (float): Classifier-free guidance weight.
        clip_min (float): Minimum value for output clipping.
        output_base_dir (str): Base directory for saving results.
        device (torch.device): Computing device (CPU or GPU).

    Returns:
        list: List of NSE results for all processed basins.
    """
    nse_results = []

    nse_csv_path = os.path.join(output_base_dir, f"{guide_w}", 'nse.csv')
    os.makedirs(os.path.dirname(nse_csv_path), exist_ok=True)

    with open(nse_csv_path, 'w') as f:
        f.write('basin,nse\n')

    for i, basin_idx in enumerate(tqdm(selected_indices, desc="Processing Basins")):
        basin_str = basin_list[basin_idx]

        Xs = load_attribute(basin_str, normal_attribute)
        Xs = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0).expand((batch_size, 27)).to(device)

        predicts = []
        observation = []

        for X, Y in testloader_list[basin_idx]:
            sample_pre = []
            X = X.to(device)

            for _ in range(num_samples):
                sample_x = sample_ddim(c=X, xs=Xs, eps_conditional=eps_conditional,
                                       eps_unconditional=eps_unconditional,
                                       n_steps=n_steps, sampling_steps=sampling_steps,
                                       clip_min=clip_min, eta=eta, guide_w=guide_w)
                sample_pre.append(sample_x.cpu().detach().numpy().flatten())

            sample_pre = np.array(sample_pre).T
            observation.append(Y.cpu().detach().numpy().flatten())
            predicts.append(sample_pre)

        predicts = np.concatenate(predicts, axis=0)
        observation = np.concatenate(observation, axis=0)
        predict_mean = np.mean(predicts, axis=1)

        output_dir = os.path.join(output_base_dir, f"{guide_w}", basin_str)
        save_data(basin_str, predicts, observation, predict_mean, output_dir)

        nse = calculate_nse(predict_mean, observation)
        nse_results.append({'basin': basin_str, 'nse': nse})

        nse_df = pd.DataFrame(nse_results)
        nse_df.to_csv(nse_csv_path, mode='a', header=False, index=False)
        nse_results = []

    return nse_results
