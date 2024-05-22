import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import datetime, time
import matplotlib.pyplot as plt
import joblib


FIB_X_TIMESERIES_DATA_FILEPATH = '../FibData/all_X_timeseries_fib.xlsx'
FIB_Y_TARGET_VARIABLE_FILEPATH = '../FibData/all_Y_timeseries_fib.xlsx'
FIB_X_STATIC_DATA_FILEPATH = '../FibData/all_X_static_fib.xlsx'

SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH = '../FibData/simple_X_timeseries_fib.xlsx'
SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH = '../FibData/simple_Y_timeseries_fib.xlsx'

FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING = '../ScaledData/all_X_timeseries_fib_after_scaling.xlsx'
FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING = '../ScaledData/all_Y_timeseries_fib_after_scaling.xlsx'
FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING = '../ScaledData/all_X_static_fib_after_scaling.xlsx'

SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING = '../ScaledData/simlpe_X_timeseries_fib_after_scaling.xlsx'
SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING = '../ScaledData/simple_Y_timeseries_fib_after_scaling.xlsx'

SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SAME_SCALING = '../ScaledData/simlpe_X_timeseries_fib_after_same_scaling.xlsx'
SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SAME_SCALING = '../ScaledData/simple_Y_timeseries_fib_after_same_scaling.xlsx'

USE_SIMPLE_FIB_DATA = False
STATIC_COLUMNS_TO_SCALE = ['fib_1', 'fib_2', 'gap_XY', 'noise_std', 'multiplier']


def load_data(filepath:str) -> pd.DataFrame:
    data = pd.read_excel(filepath)
    return data

def save_data(data:pd.DataFrame, filepath:str):
    data.to_excel(filepath, index=False)

def apply_min_max_scaler(df:pd.DataFrame, column_name:str, save_scaler:str=None) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1,1))
    if save_scaler:
        joblib.dump(scaler, save_scaler)
        print(f'Scaler saved to {save_scaler}')

    return df

def scale_data_static_data(df:pd.DataFrame) -> pd.DataFrame:
    scaled_df = df.copy()
    for column in STATIC_COLUMNS_TO_SCALE:
        scaled_df = apply_min_max_scaler(scaled_df, column)

    return scaled_df
    
def load_scaler(scaler_path:str) -> MinMaxScaler:
    return joblib.load(scaler_path)


def scale_data_timeseries_data(df:pd.DataFrame, save_scaler:str) -> pd.DataFrame:
    scaled_df = df.copy()
    scaled_df = apply_min_max_scaler(scaled_df, 'sequence', save_scaler=save_scaler)
    return scaled_df


def main():

    if USE_SIMPLE_FIB_DATA:
        # Simple data is the original fibonacci sequence data without any noise or static data

        df_x_timeseries = load_data(SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH)
        df_x_static = load_data(FIB_X_STATIC_DATA_FILEPATH)
        df_y_timeseries = load_data(SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH)

        df_x_static = scale_data_static_data(df_x_static)
        df_x_timeseries = scale_data_timeseries_data(df_x_timeseries, save_scaler='DataExtraction/Scalers/simple_x_timeseries_scaler.save')
        df_y_timeseries = scale_data_timeseries_data(df_y_timeseries, save_scaler='DataExtraction/Scalers/simple_y_timeseries_scaler.save')

        save_data(df_x_static, FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING)
        save_data(df_x_timeseries, SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING)
        save_data(df_y_timeseries, SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING)

    else:
        df_x_timeseries = load_data(FIB_X_TIMESERIES_DATA_FILEPATH)
        df_x_static = load_data(FIB_X_STATIC_DATA_FILEPATH)
        df_y_timeseries = load_data(FIB_Y_TARGET_VARIABLE_FILEPATH)
        df_x_static = scale_data_static_data(df_x_static)

        df_x_timeseries = scale_data_timeseries_data(df_x_timeseries, save_scaler='Scalers/x_timeseries_scaler.save')
        df_y_timeseries = scale_data_timeseries_data(df_y_timeseries, save_scaler='Scalers/y_timeseries_scaler.save')

        save_data(df_x_static, FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING)
        save_data(df_x_timeseries, FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING)
        save_data(df_y_timeseries, FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING)


if __name__=="__main__":
    main()