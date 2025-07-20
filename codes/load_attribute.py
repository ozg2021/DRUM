import numpy as np
import pandas as pd
from typing import Union

def load_attribute(number: Union[int, str], normal_attribute: pd.DataFrame) -> np.ndarray:
    """
    Load the attribute values for a specific basin (or ID) from the standardized attribute data.

    Args:
        number (Union[int, str]): The basin ID or name (can be an integer or string) used to select the corresponding row.
        normal_attribute (pd.DataFrame): The already standardized attribute data in a pandas DataFrame.

    Returns:
        np.ndarray: A NumPy array containing the standardized attribute values for the specified basin, with dtype float32.
    """
    attribute = normal_attribute.loc[number].values
    return attribute.astype(np.float32)
