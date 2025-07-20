"""
Predefined mean and standard deviation values for different forcing sources and date ranges.
These values are used for data normalization during preprocessing.
"""

PREDEFINED_MEAN_STD_VALUES = {
    "1980-10-01_to_1990-09-30": {
        'daymet': {
            'prcp(mm/day)': {'mean': 3.3071364966800942, 'std': 7.684093525598007},
            'srad(W/m2)': {'mean': 342.5638361355024, 'std': 133.59504091822393},
            'tmax(C)': {'mean': 16.557623828647916, 'std': 11.196600755310207},
            'tmin(C)': {'mean': 3.8404167362825734, 'std': 10.385570052590385},
            'vp(Pa)': {'mean': 949.3694139165813, 'std': 651.9134570699803}
        },
        'nldas': {
            'prcp(mm/day)': {'mean': 3.1332692506028224, 'std': 7.481922345634662},
            'srad(W/m2)': {'mean': 352.2189134194712, 'std': 128.26893859053914},
            'tmax(C)': {'mean': 10.557373226857102, 'std': 11.007000809229826},
            'tmin(C)': {'mean': 10.557373226857102, 'std': 11.007000809229826},
            'vp(Pa)': {'mean': 1066.4870666229376, 'std': 706.2899735318335}
        },
        'maurer': {
            'prcp(mm/day)': {'mean': 3.171398274144343, 'std': 6.778514620196002},
            'srad(W/m2)': {'mean': 374.43500524439804, 'std': 130.6893022239598},
            'tmax(C)': {'mean': 9.970365761969296, 'std': 10.900710074762467},
            'tmin(C)': {'mean': 9.970365761969296, 'std': 10.900710074762467},
            'vp(Pa)': {'mean': 909.8520584856116, 'std': 618.6057375426647}
        }
    }
}


def get_date_range_key(start_date_str: str, end_date_str: str) -> str:
    """
    Generate a date range key from start and end date strings.

    Args:
        start_date_str: Start date in format 'YYYY-MM-DDTHH:MM:SS'
        end_date_str: End date in format 'YYYY-MM-DDTHH:MM:SS'

    Returns:
        Date range key in format 'YYYY-MM-DD_to_YYYY-MM-DD'
    """
    import dateutil.parser

    start_date = dateutil.parser.parse(start_date_str)
    end_date = dateutil.parser.parse(end_date_str)

    return f"{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}"


def get_mean_std_values(date_ranges: dict) -> dict:
    """
    Get predefined mean and std values for the given date range.

    Args:
        date_ranges: Dictionary containing 'train_start' and 'train_end' keys

    Returns:
        Dictionary containing mean and std values for the date range

    Raises:
        ValueError: If the date range is not found in predefined values
    """
    if not date_ranges or 'train_start' not in date_ranges or 'train_end' not in date_ranges:
        raise ValueError("date_ranges must contain 'train_start' and 'train_end' keys")

    date_range_key = get_date_range_key(date_ranges['train_start'], date_ranges['train_end'])

    if date_range_key not in PREDEFINED_MEAN_STD_VALUES:
        available_ranges = list(PREDEFINED_MEAN_STD_VALUES.keys())
        raise ValueError(
            f"Date range '{date_range_key}' not found in predefined values.\n"
            f"Available date ranges: {available_ranges}\n"
            f"Please add the required date range to PREDEFINED_MEAN_STD_VALUES in model/predefined_stats.py"
        )

    return PREDEFINED_MEAN_STD_VALUES[date_range_key]
