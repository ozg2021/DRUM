import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from codes.create_data import create_X, create_Y, TimeseriesDataset
from codes.load_forcing import load_forcing
from codes.load_discharge import load_discharge
from codes.check_basin_start_date import check_basin_start_date
from codes.predefined_stats import get_mean_std_values


def pre_data(
        time_step: int,
        basin_list: List[str],
        is_train: bool = True,
        is_valid: bool = False,
        date_ranges: Optional[Dict[str, str]] = None,
        batch_size: int = 256,
        camels_path: Path = Path('CAMELS'),
        forcing_sources: Dict[str, List[str]]=None
) -> Tuple[List[DataLoader], np.ndarray]:
    """
    Prepares the training or validation datasets by loading the meteorological forcing data
    and discharge data for each basin, processing the data, and returning DataLoaders for training.

    :param time_step: Time step used for creating time series data (e.g., number of previous time steps).
    :param basin_list: List of basin names or IDs to load data for.
    :param is_train: Flag indicating whether to prepare the training dataset (default is True).
    :param is_valid: Flag indicating whether to prepare the validation dataset (default is False).
    :param date_ranges: Dictionary containing the start and end dates for the training, validation, or test sets.
    :param batch_size: The batch size for the DataLoader (default is 256).
    :param camels_path: Path to the CAMELS dataset (default is 'autodl-tmp/data/CAMELS').
    :param forcing_sources: LDictionary where keys are data source names, and values are lists of variables to load for each source.
    :return: A tuple containing a list of DataLoaders (one per basin) and a NumPy array with the standard deviations of discharge values for each basin.
    """
    try:
        mean_std_values = get_mean_std_values(date_ranges)
        print(f"Using predefined mean/std values for date range: {date_ranges['train_start']} to {date_ranges['train_end']}")
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit("Please add the required date range configuration in codes/predefined_stats.py")

    # print(mean_std_values)

    data_list = []
    std1 = []

    data_dir = camels_path / 'usgs_streamflow'

    for i in tqdm(basin_list):
        start_date = check_basin_start_date(i, data_dir)
        if start_date is None:
            print(f"Could not find start date for basin {i}")
            continue

        adjusted_date_ranges = date_ranges.copy() if date_ranges else {}
        if start_date != pd.Timestamp('1980-01-01'):
            adjusted_date_ranges['train_start'] = start_date.strftime('%Y-%m-%dT00:00:00')

        forcing, area = load_forcing(camels_path, i, is_train, is_valid, adjusted_date_ranges,
                                     forcing_sources, mean_std_values)
        X = forcing

        Y = load_discharge(camels_path, i, area, is_train, is_valid, adjusted_date_ranges)

        X = np.array(create_X(X, time_step))
        Y = np.array(create_Y(Y, time_step))

        neg_idx = np.where(Y < 0)[0]
        if len(neg_idx) > 0:
            print(f"Warning: Basin {i} contains {len(neg_idx)} negative discharge samples. Removing.")
            X = np.delete(X, neg_idx, axis=0)
            Y = np.delete(Y, neg_idx, axis=0)

        nan_idx = np.where(np.isnan(Y))[0]
        if len(nan_idx) > 0:
            print(f"Warning: Basin {i} contains {len(nan_idx)} NaN discharge samples. Removing.")
            X = np.delete(X, nan_idx, axis=0)
            Y = np.delete(Y, nan_idx, axis=0)

        if len(Y) == 0:
            print(f"Error: Basin {i} has no valid data after cleaning. Skipping this basin.")
            continue

        dataset = TimeseriesDataset(X, Y)

        if is_train:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        data_list.append(train_loader)
        std = np.std(Y)
        std1.append(std)

    return data_list, np.array(std1)
