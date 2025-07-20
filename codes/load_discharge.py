import pandas as pd
import numpy as np
import dateutil.parser
from pathlib import Path
from typing import Dict, Optional


def load_discharge(
        discharge_root: Path,
        basin: str,
        area: float,
        is_train: bool = True,
        is_valid: bool = False,
        date_ranges: Optional[Dict[str, str]] = None
) -> np.ndarray:
    """
    Loads discharge data for the specified basin and returns normalized discharge values.

    Args:
        discharge_root (Path): Path to the root directory containing discharge data.
        basin (str): Name of the basin, used to find the corresponding file.
        area (float): Area of the basin in square kilometers.
        is_train (bool, optional): Whether to load training data. Defaults to True.
        is_valid (bool, optional): Whether to load validation data. Defaults to False.
        date_ranges (Optional[Dict[str, str]], optional): Dictionary specifying the start and end dates
                                                          for the train/validation/test splits.

    Returns:
        np.ndarray: Standardized discharge data in mm/day per square kilometer.
    """
    discharge_path = discharge_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))

    basin = ''.join(basin)

    file_path = [f for f in files if f.name[:8] == basin]

    if len(file_path) == 0:
        raise RuntimeError(f'No file found for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'year', 'mnth', 'day', 'qobs', 'flag']

    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    df.columns = [col.lower() for col in df.columns]

    dates = (df['year'].map(str) + "/" + df['mnth'].map(str) + "/" + df['day'].map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    if is_train:
        if date_ranges:
            df = df[dateutil.parser.parse(date_ranges["train_start"]):dateutil.parser.parse(date_ranges["train_end"])]
    elif is_valid:
        if date_ranges:
            df = df[dateutil.parser.parse(date_ranges["valid_start"]):dateutil.parser.parse(date_ranges["valid_end"])]
    else:
        if date_ranges:
            df = df[dateutil.parser.parse(date_ranges["test_start"]):dateutil.parser.parse(date_ranges["test_end"])]

    discharge = df['qobs'].values.astype(np.float32)
    discharge = 28316846.592 * discharge * 86400 / (area * 10 ** 6)

    return discharge
