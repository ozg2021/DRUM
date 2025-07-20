import os
import torch
import numpy as np
from tqdm import tqdm


def get_current_lr(optimizer):
    """
    Extract the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object.

    Returns:
        float: Current learning rate of the first parameter group.
    """
    return optimizer.param_groups[0]['lr']


def train_conditional(dataloader_list, save_dir, eps_model, diffusion, optimizer, static_attributes, epochs,
                      lr_schedule, logger, device):
    """
    Train the conditional diffusion model for noise prediction.

    This function trains a conditional model that uses both meteorological forcing data
    and static basin characteristics to predict noise in the diffusion process.

    Args:
        dataloader_list (list): List of DataLoader objects, each corresponding to a basin's training data.
                               Each DataLoader contains (meteorological_data, target_runoff) pairs.
        save_dir (str): Directory path where model checkpoints will be saved after each epoch.
        eps_model (nn.Module): The noise prediction model (EpsModel) being trained.
        diffusion (object): The diffusion model object that computes the conditional loss.
        optimizer (torch.optim.Optimizer): The optimizer used for training (e.g., Adam, AdamW).
        static_attributes (list): List of static feature tensors for each basin (e.g., topography, soil properties).
                                 Length should match len(dataloader_list).
        epochs (int): Total number of training epochs to run.
        lr_schedule (list): List of tuples (epoch, new_lr) for learning rate scheduling.
                           Example: [(10, 1e-4), (20, 1e-5)] adjusts LR at epochs 10 and 20.
        logger (logging.Logger): Logger object for recording training progress and metrics.
        device (torch.device): Computing device (CPU or GPU) for tensor operations.

    Returns:
        list: Training losses for each epoch (averaged over all batches and basins).
    """
    train_loss = []

    for epoch in range(epochs):
        epoch_loss = []

        for schedule_epoch, new_lr in lr_schedule:
            if epoch == schedule_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f'Epoch {epoch + 1}/{epochs}, Learning Rate adjusted to: {new_lr:.1e}')
                break

        for basin_idx in tqdm(range(len(dataloader_list)), desc=f'Training Epoch {epoch + 1}/{epochs}'):
            for rain, runoff in dataloader_list[basin_idx]:
                optimizer.zero_grad()

                rain = rain.to(device)
                runoff = runoff.to(device)

                xs = static_attributes[basin_idx].to(device)

                loss = diffusion.loss_conditional(runoff, rain, xs)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(eps_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        train_loss.append(avg_loss)

        save_path = os.path.join(save_dir, f'eps_model_epoch_{epoch + 1}.pt')
        torch.save(eps_model.state_dict(), save_path)

        current_lr = get_current_lr(optimizer)
        logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Learning Rate: {current_lr:.1e}')

    return train_loss


def train_unconditional(dataloader_list, save_dir, eps_model, diffusion, optimizer, static_attributes, epochs,
                        lr_schedule, logger, device):
    """
    Train the unconditional diffusion model for noise prediction.

    This function trains an unconditional model that uses only static basin characteristics
    to predict noise in the diffusion process, without meteorological forcing data.
    This is used for classifier-free guidance during inference.

    Args:
        dataloader_list (list): List of DataLoader objects, each corresponding to a basin's training data.
                               Each DataLoader contains (meteorological_data, target_runoff) pairs.
                               Note: meteorological_data is ignored in unconditional training.
        save_dir (str): Directory path where model checkpoints will be saved after each epoch.
        eps_model (nn.Module): The noise prediction model (EpsModel) being trained.
        diffusion (object): The diffusion model object that computes the unconditional loss.
        optimizer (torch.optim.Optimizer): The optimizer used for training (e.g., Adam, AdamW).
        static_attributes (list): List of static feature tensors for each basin (e.g., topography, soil properties).
                                 Length should match len(dataloader_list).
        epochs (int): Total number of training epochs to run.
        lr_schedule (list): List of tuples (epoch, new_lr) for learning rate scheduling.
                           Example: [(10, 1e-4), (20, 1e-5)] adjusts LR at epochs 10 and 20.
        logger (logging.Logger): Logger object for recording training progress and metrics.
        device (torch.device): Computing device (CPU or GPU) for tensor operations.

    Returns:
        list: Training losses for each epoch (averaged over all batches and basins).
    """
    train_loss = []

    for epoch in range(epochs):
        epoch_loss = []

        for schedule_epoch, new_lr in lr_schedule:
            if epoch == schedule_epoch:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f'Epoch {epoch + 1}/{epochs}, Learning Rate adjusted to: {new_lr:.1e}')
                break

        for basin_idx in tqdm(range(len(dataloader_list)), desc=f'Unconditional Training Epoch {epoch + 1}/{epochs}'):
            for rain, runoff in dataloader_list[basin_idx]:
                optimizer.zero_grad()

                runoff = runoff.to(device)

                xs = static_attributes[basin_idx].to(device)

                loss = diffusion.loss_unconditional(runoff, xs)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(eps_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss.append(loss.item())

        avg_loss = np.mean(epoch_loss)
        train_loss.append(avg_loss)

        save_path = os.path.join(save_dir, f'eps_model_epoch_{epoch + 1}.pt')
        torch.save(eps_model.state_dict(), save_path)

        current_lr = get_current_lr(optimizer)
        logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Learning Rate: {current_lr:.1e}')

    return train_loss
