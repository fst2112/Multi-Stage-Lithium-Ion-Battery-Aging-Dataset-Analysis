import numpy as np
import pandas as pd
from pathlib import Path


def import_datafile(file: Path) -> pd.DataFrame:
    """
    Import raw data files to a dataframe and convert timestamp to seconds

    :param file: Path: path to the file
    :type file: Path

    """

    # read file
    df = pd.read_csv(file, encoding='iso-8859-1')

    # timestamp to seconds
    first_run_time = df.run_time[0]
    if first_run_time is not None and ':' in str(first_run_time):
        df.run_time = timestamp_to_seconds(df.run_time)

    # reset time to start with 0
    first_run_time = df.run_time[0]
    df.run_time = df.run_time - first_run_time

    return df


def timestamp_to_seconds(timestamp: np.ndarray) -> np.ndarray:
    """
    Convert timestamp column (hh:mm:SS.sss) to seconds

    :param timestamp: np.ndarray: array of BaPoBs timestamp hh:mm:SS.sss
    :return: np.ndarray: array of times in second

    """
    try:
        seconds = pd.to_timedelta(timestamp)
        seconds = seconds / np.timedelta64(1, 's')
    except ValueError:
        print("Error: The input timestamp is not in the correct format (hh:mm:SS.sss).")
        return None

    return seconds
