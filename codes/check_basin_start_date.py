import os
import pandas as pd
from pathlib import Path
from typing import Optional


def check_basin_start_date(basin: str, data_dir: Path) -> Optional[pd.Timestamp]:
    """
    Checks the start date of the specified basin.

    Searches through the specified directory for files that start with the given basin name,
    reads the first line of the file (assuming it contains the date in the format 'YYYY MM DD'),
    and returns the date.

    :param basin: Basin name used to match the file prefix.
    :param data_dir: Directory path where the basin data files are located.
    :return: A 'pd.Timestamp' object representing the start date of the basin if found;
             otherwise, returns 'None'.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Data directory {data_dir} is invalid or does not exist.")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.startswith(basin):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()

                    if not first_line:
                        print(f"Warning: First line is empty in file {file_path}")
                        continue

                    parts = first_line.split()
                    if len(parts) >= 4:
                        date_str = '-'.join(parts[1:4])
                        date = pd.to_datetime(date_str, format='%Y-%m-%d')
                        return date
                    else:
                        print(f"Warning: Insufficient date parts in file {file_path}. Expected at least 4 parts, got {len(parts)}")

                except FileNotFoundError:
                    print(f"Error: File {file_path} not found")
                except PermissionError:
                    print(f"Error: Permission denied when reading file {file_path}")
                except pd.errors.ParserError as e:
                    print(f"Error: Failed to parse date in file {file_path}: {e}")
                except Exception as e:
                    print(f"Error occurred while reading file {file_path}: {e}")

    return None
