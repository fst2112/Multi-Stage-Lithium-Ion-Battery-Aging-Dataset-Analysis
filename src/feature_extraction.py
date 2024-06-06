import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.interpolate import griddata, CubicSpline
from scipy.signal import savgol_filter


############################################################################################################
# CAPACITY EXTRACTION
############################################################################################################
def df_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting the capacity measurement from larger DataFrame

    :param df:
    :return df_capa: pd.DataFrame: DataFrame containing capacity measurement only
    """
    df_capa_meas = df[((df.step_type == 21) & (df.c_cur > 0)) | ((df.step_type == 22) & (df.c_cur < 0))]

    return df_capa_meas


def capacity(df: pd.DataFrame) -> list:
    """
    Evaluate capacity of measurement in df with step_type = [21,22]

    :param df: pd.DataFrame: df with step_type = [21,22]
    :return: {'Q_mean': float, 'Q_ch': float, 'Q_dch': float, 'q_ch': np.ndarray, 'q_dch': np.ndarray}
    """
    q_kapa_ch = q_calc(df[(df.step_type == 21) & (df.c_cur > 0)])
    q_kapa_dch = q_calc(df[(df.step_type == 22) & (df.c_cur < 0)])

    capa_ch = q_kapa_ch[-1] - q_kapa_ch[0]
    capa_dch = q_kapa_dch[0] - q_kapa_dch[-1]

    capa_mean = (capa_ch + capa_dch) / 2 / 3600
    capa_ch = capa_ch / 3600
    capa_dch = capa_dch / 3600

    return {'Q_mean': capa_mean, 'Q_ch': capa_ch, 'Q_dch': capa_dch, 'q_ch': q_kapa_ch, 'q_dch': q_kapa_dch}


def q_calc(df: pd.DataFrame) -> np.ndarray:
    """
    Integrate current over time to get charge throughput

    :param df:
    :return q: np.array:
    """
    try:
        q_val = (df.run_time.diff() * df.c_cur).fillna(0).values
        q = np.cumsum(q_val)

        return q

    except Exception as e:
        print(f'Exception: q_calc: {e}')
        q = np.array([np.nan])
        return q


############################################################################################################
# OPEN CIRCUIT VOLTAGE (OCV) EXTRACTION
############################################################################################################
def df_ocv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting the ocv measurement from larger DataFrame

    :param df: pd.DataFrame: larger dataframe containing ocv measurement
    :return df_ocv_meas: pd.DataFrame: DataFrame containing ocv measurement only
    """

    df_ocv_meas = df[(df.step_type == 31) | (df.step_type == 32)]

    return df_ocv_meas


def ocv_curve(df: pd.DataFrame, enable_filter: bool = False, polyorder: int = 3) -> dict:
    """
    Extracting the open circuit voltage curve (OCV) contained in the dataframe with step_type = [31,32].
    Here, only the constant current CC-OCV measurement can be evaluated.

    :param polyorder: Order of the polynomial filter for smoothing the OCV curve
    :type polyorder: int
    :param enable_filter: Enable filter for smoothing the OCV curve
    :type enable_filter: bool
    :param df:
    :return: {'SoC': np.ndarray, 'OCV': np.ndarray, 'OCV_ch': np.ndarray, 'OCV_dch': np.ndarray}
    """
    # Select the charging (CC) / discharging areas and store as vectors
    c_cur_dch = df.c_cur[(df.step_type == 32)].values
    c_vol_dch = df.c_vol[(df.step_type == 32)].values

    c_cur_ch = df.c_cur[(df.step_type == 31)].values
    c_vol_ch = df.c_vol[(df.step_type == 31)].values

    # Time section division into soc-interval
    soc_dch = np.transpose(np.linspace(100, 0, (len(c_cur_dch))))
    soc_ch = np.transpose(np.linspace(0, 100, (len(c_cur_ch))))

    # Interpolation of the charging and discharging curve
    soc = np.linspace(0, 100, 1001)
    ocv_dch = griddata(soc_dch, c_vol_dch, soc, method="nearest")
    ocv_ch = griddata(soc_ch, c_vol_ch, soc, method="nearest")

    if enable_filter:
        window_size = int(len(ocv_dch) * 0.001)
        ocv_dch = savgol_filter(ocv_dch, window_size, polyorder)
        ocv_ch = savgol_filter(ocv_ch, window_size, polyorder)

    # Formation of the open circuit voltage curve (average)
    ocv = (ocv_dch + ocv_ch) / 2

    return {'SOC': soc.round(2), 'OCV': ocv.round(4), 'OCV_ch': ocv_ch.round(4), 'OCV_dch': ocv_dch.round(4)}


def dva_curve(ocv: np.ndarray, soc: np.ndarray, average_window: int = 10, enable_spline: bool = False) -> np.ndarray:
    """
    Differentiate the OCV-curve to get the DVA-curve

    :param ocv: Array of OCV values
    :type ocv: np.ndarray
    :param soc: Array of SoC values
    :type soc: np.ndarray
    :param average_window: int = 10: window for moving average
    :param enable_spline: Enable spline interpolation
    :type enable_spline: bool
    :return:
    """
    dva = moving_average(data=np.diff(ocv), window=average_window)

    if enable_spline:
        soc_vec = np.linspace(min(soc), max(soc), 1000)
        ocv_spline = CubicSpline(soc, ocv)
        ocv_vec = ocv_spline(soc_vec)
        dva = savgol_filter(np.diff(ocv_vec), 51, 3)

    return dva.round(8)


############################################################################################################
# DC INNER RESISTANCE (DCIR) EXTRACTION
############################################################################################################

def df_single_pulse(df: pd.DataFrame,
                    step_type: int = 5032,
                    extend_pulse: list = None,
                    ) -> pd.DataFrame:
    """
    Extract single pulse from pulse test contained in df with extended time window before and
    after pulse start and stop.

    :param df: pd.DataFrame: df containing pulse measurements
    :type df: pd.DataFrame
    :param step_type: int = 5032: define the step_type to extract. Default: 50% SoC, -1C (discharge)
    :type step_type: int
    :param extend_pulse: list = None: extend pulse by [start_t_s, stop_t_s] seconds before and after step_type
    :type extend_pulse: list
    :return: df: pd.Dataframe: Dataframe containing just the wanted pulse
    """

    if extend_pulse is None:
        extend_pulse = [0, 0]

    # extract additional time before and after pulse
    t_before, t_after = extend_pulse

    # Limit the values to the range [0, 540] for t_before and [0, 30] for t_after
    t_before = max(0, min(t_before, 540))
    t_after = max(1, min(t_after, 30))

    # find start and stop index of pulse in df
    start_idx = df[df.step_type == step_type].index[0]
    stop_idx = df[df.step_type == step_type].index[-1]

    # find time of pulse start and stop in df
    pulse_start_time = df.loc[start_idx, 'run_time']
    pulse_end_time = df.loc[stop_idx, 'run_time']
    # shift start and stop index accordingly to extend_pulse values
    start_idx = df[df['run_time'] >= (pulse_start_time - t_before - 1)].index[0]
    stop_idx = df[df['run_time'] <= (pulse_end_time + t_after)].index[-1]

    # extract pulse with extended time window from df and reset index
    df = df.copy()[start_idx:stop_idx].reset_index(drop=True)

    # identify state changes by current changes
    current_changes_idx = abs(df.c_cur.diff()) > 1
    current_changes_idx = current_changes_idx.index[current_changes_idx == True]

    if len(current_changes_idx) < 2:
        print('Warning: Pulse contains less than 2 current changes. Pulse might be corrupted.')

    return df


def rdc_extract(df: pd.DataFrame, ocv_fcns: dict, t: float = 10) -> list:
    """
    Extract RDC after t seconds from pulse in Ohm.
    OCV functions are needed for OCV correction. The functions are stored in a dictionary:
    ocv_fcns = {'f_ocv(capacity)': interpolate.interp1d(self.df_ocv_ref['capacity'].values,
                                                                 self.df_ocv_ref['mean'].values,
                                                                 kind='linear'),
                         'f_capacity(ocv)': interpolate.interp1d(self.df_ocv_ref['mean'].values,
                                                                 self.df_ocv_ref['capacity'].values,
                                                                 kind='linear')
                         }

    :param df: pd.DataFrame: results from df_single_pulse
    :type df: pd.DataFrame
    :param ocv_fcns: dict: dictionary containing the functions f_ocv(capacity) and f_capacity(ocv)
    :type ocv_fcns: dict
    :param t: int = 10: reference time  in seconds for RDC
    :type t: float
    :return: dict: {'RDC': float, 'I_pulse': float}: RDC[Ohm], I_pulse[A]
    """

    # find index triggered by a voltage change > abs(0.005V)
    c_vol0 = df.c_vol[0]
    c_vol_threshold = 0.005
    c_vol_trigger_index = ((df['c_vol'] - c_vol0).abs() >= c_vol_threshold).idxmax()

    # extract voltage before the pulse by taking the median of the values before voltage change
    c_vol0 = np.median(df.c_vol[:c_vol_trigger_index - 1])
    c_capa0 = ocv_fcns['f_capacity(ocv)'](c_vol0)

    # Extract the pulse only by identifying state changes by current changes
    current_changes_idx = abs(df.c_cur.diff()) > 1
    current_changes_idx = current_changes_idx.index[current_changes_idx == True]

    pulse_idx_0 = current_changes_idx[0]
    pulse_idx_1 = current_changes_idx[-1]

    df = df.copy()[pulse_idx_0 - 1:pulse_idx_1].reset_index(drop=True)

    # reset time base
    df.run_time = df.run_time.copy() - df.run_time.copy()[0]

    # Check the pulse duration
    if df.iloc[-1]["run_time"] < t:
        return {'RDC': np.nan, 'I_pulse': np.nan}

    # calculate charge throughput in Ah during pulse
    df['q'] = q_calc(df) / 3600

    # find index of value at t seconds
    time_end_index = np.where(df.run_time <= t)[0][-1]

    # find voltage at t seconds after pulse start including ocv correction
    c_capa1 = df.q[time_end_index] + c_capa0
    c_vol1_ocv_delta = ocv_fcns['f_ocv(capacity)'](c_capa1) - c_vol0
    c_vol1 = df.c_vol[time_end_index] - c_vol1_ocv_delta

    # calculate pulse current
    c_cur = np.median(df.c_cur[:time_end_index])

    # Check for CV phase at the end of the pulse
    if abs(df.iloc[time_end_index-10]["c_cur"]) < abs(0.9*c_cur):
        return {'RDC': np.nan, 'I_pulse': np.nan}

    # calculate rdc
    rdc = abs((c_vol1 - c_vol0) / c_cur)

    return {'RDC': rdc, 'I_pulse': c_cur}


############################################################################################################
# CALENDAR TIME EXTRACTION
############################################################################################################
def calendar_time(meta0: Path,
                  meta1: Path,
                  df_cu0: pd.DataFrame
                  ) -> float:
    """
    Find the calendar time of the measurement in the experimental campaign.
    This is done by comparing the date of the measurement with the date of the previous measurement.
    1. Find the date of the measurement.
    2. Find the date of the previous measurement.
    3. Extract the calendar time of previous CU and exCU measurements.
    4. Calculate the calendar time of the measurement
    5. Correct the calendar time by the number of days, when CUs and exCUs were performed.

    :param meta0: str: metadata of the previous measurement
    :param meta1: str: metadata of the current measurement
    :param df_cu0: pd.DataFrame: dataframe of the previous measurement

    :return: cal_time: float: calendar time of the aging phase before the CU and after the previous CU
    """
    # total time in days between the starts of the measurements
    total_time = (date_extract(meta1) - date_extract(meta0)).days
    # CU time in seconds to days
    cu_time = df_cu0.run_time.iloc[-1]/3600/24
    # calendar time in days
    cal_time: float = total_time - cu_time

    return cal_time


def date_extract(file: Path) -> datetime:
    # extract the date of measurement of the dataframe from the metadata
    with open(file, "r") as f:
        content = f.readlines()
    for line in content:
        if "Measurement start date" in line:
            date = line.split(":")[1].strip()

    try:
        date = datetime.strptime(date, '%d.%m.%y').strftime('%d.%m.%y')
    except ValueError:
        date = datetime.strptime(date, '%d.%m.%Y').strftime('%d.%m.%y')

    return datetime.strptime(date, '%d.%m.%y').date()


def cycle_time(df: pd.DataFrame) -> float:
    """
    Compute the time of cycling

    :param df: pd.DataFrame: cycling data
    :type df: pd.DataFrame
    :return: t: float: time in days
    """
    try:
        cyc_time = np.sum(df.run_time_diff.values) / (3600 * 24)  # t in days

    except Exception as e:
        print(f'Exception: cycle_time: {e}')
        cyc_time = np.nan

    return cyc_time


############################################################################################################
# FULL EQUIVALENT CYCLE (FEC) EXTRACTION#
############################################################################################################
def df_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding df[run_time_diff] and extracting the cycles from cycle measurement DataFrame

    :param df: pd.DataFrame
    :return df_cycles: pd.DataFrame: DataFrame containing cycle data only
    """
    df['run_time_diff'] = df.run_time.copy().diff()
    df_cycles = df[((df.step_type == 41) | (df.step_type == 42)) & (abs(df.c_cur) > 0.05)].reset_index(drop=True)

    return df_cycles


def fec_extract(df: pd.DataFrame, capa_ref: float = 4.9) -> float:
    """
    Calculate Full Equivalent Cycles (FEC) from cycling data.

    :param df: pd.DataFrame: cycling data
    :type df: pd.DataFrame
    :param capa_ref: float: capacity, the FECs are referenced to
    :type capa_ref: float
    :return: fec
    """
    fec: float

    try:
        q_pos_half = df[df.step_type == 41]
        q_neg_half = df[df.step_type == 42]

        q_val_pos = (q_pos_half.run_time_diff * q_pos_half.c_cur).fillna(0).values
        q_val_neg = (q_neg_half.run_time_diff * q_neg_half.c_cur).fillna(0).values

        fec_pos = abs(sum(q_val_pos)).round(2)
        fec_neg = abs(sum(q_val_neg)).round(2)

        fec_temp = fec_pos + fec_neg

        fec = fec_temp / (2 * capa_ref * 3600)

        return round(fec, 4)

    except (Exception,) as e:
        print(f'Exception: fec_extract: {e}')
        fec = np.nan
        return fec


############################################################################################################
# MOVING AVERAGE FILTER
############################################################################################################
def moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    smooth measurements by applying moving average filter

    :param window: int
    :param data: np.ndarray
    :return: np.ndarray

    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


