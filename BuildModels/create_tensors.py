# This script is used to create tensors from the dataframes. It is not meant to 
# be run as a standalone script. The main function is used to get the training and test data.
# These functions are called from the create_[model].py scripts.

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
from tensorflow import keras
from typing import Tuple

FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING = '../ScaledData/all_X_timeseries_fib_after_scaling.xlsx'
FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING = '../ScaledData/all_Y_timeseries_fib_after_scaling.xlsx'
FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING = '../ScaledData/all_X_static_fib_after_scaling.xlsx'

SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH = '../FibData/simple_X_timeseries_fib.xlsx'
SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH = '../FibData/simple_Y_timeseries_fib.xlsx'

USE_SIMPLE_FIB_DATA = True
TEST_SIZE = 0.2
SEQUENCE_LENGTH = 10


def load_data(filepath:str) -> pd.DataFrame:
    data = pd.read_excel(filepath)
    return data

def reshape_data_to_tensors(
    X:pd.DataFrame, 
    y:pd.DataFrame, 
    convert_to_tensor=True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reshape the data to be 3D (samples x timesteps x features) 
    and convert to tensors.
    Still includes ids, they have not been dropped, they are needed to reshape the data.
    They will be dropped here.

    Parameters:
        convert_to_tensor: if False, only reshapes data

    Returns:
        X, y: the reshaped data ( number of samples, number of timesteps, number of features)    
    """

    id_column = 'id'
    len_unique_patients_X = 0
    try:
        len_unique_patients_X = len(X[id_column].unique())
        len_unique_patients_y = len(y[id_column].unique())
        X = X.drop(columns=id_column)
        y = y.drop(columns=id_column)
    except KeyError:
        print("id not a column")
        raise SystemExit()

    X_reshape = np.reshape(X.to_numpy(), (len_unique_patients_X, SEQUENCE_LENGTH, X.shape[1]))
    y_reshape = np.reshape(y.to_numpy(), (len_unique_patients_y, SEQUENCE_LENGTH, y.shape[1]))
    if convert_to_tensor:
        X_tensor = tf.convert_to_tensor(X_reshape)
        y_tensor = tf.convert_to_tensor(y_reshape)

    return X_tensor, y_tensor


def pad_sequences_to_equal_measurements(data:pd.DataFrame, max_sequence_length:int=SEQUENCE_LENGTH, pad_value:int=-1) -> pd.DataFrame:
    """
    (Pre)Pads sequences that are to short, (Pre) truncates sequences that are too long.
    Pre can be changed to post.
    For the Fib experiment, we dont need this as all sequences are same lenghts, unlike the Sepsis or CF data. 
    
    Parameters:
        data: The processed data, with non equal number of measurements per patient.
        max_sequence_length: number of measurements per patient to return.

    Returns:
        The new data, where each patient only has sample_size number of measurements. Includes patient number. 
    """
    unique_patients = []
    try:
        unique_patients = data['id'].unique()
    except KeyError as e:
        logging.error('PATNUM not a column: %s', e)
        raise SystemExit()

    df_all_patients = pd.DataFrame(columns=data.columns)
    for patient in unique_patients:
        patient_data = data[data['id'] == patient].drop(columns = ['id'])
        df_all_features_patient = pd.DataFrame()

        for column in patient_data.columns:
            np_feature = patient_data[column].to_numpy()
            np_feature = np.reshape(np_feature, (1, len(np_feature)))
            padded_sequence = keras.preprocessing.sequence.pad_sequences(np_feature, 
                                                                         maxlen=max_sequence_length,
                                                                         padding='pre', truncating='pre',
                                                                         dtype='float32', 
                                                                         value=pad_value).flatten()

            df_all_features_patient[column] = pd.Series(padded_sequence)
            
        df_all_features_patient['id'] = patient
        df_all_patients = df_all_patients.append(df_all_features_patient)
    
    df_all_patients = pd.DataFrame(df_all_patients)
    return df_all_patients


def get_training_and_test_data(timeseries_df:pd.DataFrame, static_df:pd.DataFrame, target_df:pd.DataFrame, return_padded_tensors=False, return_val_data=False):
    '''
    Gets the static and timeseries data.
    splits the sequences into training, validation and test sequences.

    All sequences have the same length.
    Convert to tensors.
    We only have X static data, but have X and Y timeseries data.
    The static data are returned as dataframes, the timeseries data as tensors if return_padded_tensors=True, else as dataframes.
    if return_val_data=True, the function returns validation data as well.

    Returns:
        (X_train_static, X_train_timeseries, y_train_static, y_train_timeseries),
        (X_test_static, X_test_timeseries, y_test_static, y_test_timeseries
    '''

    id_column = 'id'
    target_variable = 'sequence'

    unique_sequence_ids = static_df[id_column].unique()
    train_sequence_ids, test_sequences_ids = train_test_split(unique_sequence_ids, test_size=TEST_SIZE, random_state=42)
    train_sequence_ids, val_sequences_ids = train_test_split(train_sequence_ids, test_size=0.17, random_state=42)

    static_train_data = static_df[static_df[id_column].isin(train_sequence_ids)]
    static_test_data = static_df[static_df[id_column].isin(test_sequences_ids)]
    static_val_data = static_df[static_df[id_column].isin(val_sequences_ids)]

    X_train_timeseries = timeseries_df[timeseries_df[id_column].isin(train_sequence_ids)]
    X_test_timeseries = timeseries_df[timeseries_df[id_column].isin(test_sequences_ids)]
    X_val_timeseries = timeseries_df[timeseries_df[id_column].isin(val_sequences_ids)]

    y_train_timeseries = target_df[target_df[id_column].isin(train_sequence_ids)]
    y_test_timeseries = target_df[target_df[id_column].isin(test_sequences_ids)]
    y_val_timeseries = target_df[target_df[id_column].isin(val_sequences_ids)]

    # drop id column from static data
    X_train_static = static_train_data.drop([id_column], axis=1)
    X_test_static = static_test_data.drop([id_column], axis=1)
    X_val_static = static_val_data.drop([id_column], axis=1)

    # keep only the 'sequence' column in the timerseries data
    X_train_timeseries = X_train_timeseries[[id_column, target_variable]]
    X_test_timeseries = X_test_timeseries[[id_column, target_variable]]
    X_val_timeseries = X_val_timeseries[[id_column, target_variable]]

    y_train_timeseries = y_train_timeseries[[id_column, target_variable]]
    y_test_timeseries = y_test_timeseries[[id_column, target_variable]]
    y_val_timeseries = y_val_timeseries[[id_column, target_variable]]

    # convert static data to tensors
    X_train_static = tf.convert_to_tensor(X_train_static)
    X_val_static = tf.convert_to_tensor(X_val_static)
    X_test_static = tf.convert_to_tensor(X_test_static)

    # convert timeseries data to tensors
    X_train_timeseries, y_train_timeseries = reshape_data_to_tensors(X_train_timeseries, y_train_timeseries)
    X_test_timeseries, y_test_timeseries = reshape_data_to_tensors(X_test_timeseries, y_test_timeseries)
    X_val_timeseries, y_val_timeseries = reshape_data_to_tensors(X_val_timeseries, y_val_timeseries)

    if return_val_data:
        return (X_train_static, X_train_timeseries, y_train_timeseries ), (X_test_static, X_test_timeseries, y_test_timeseries), (X_val_static, X_val_timeseries, y_val_timeseries)
    
    return (X_train_static, X_train_timeseries, y_train_timeseries), (X_test_static, X_test_timeseries, y_test_timeseries)


def main():

    if USE_SIMPLE_FIB_DATA:
        df_x_timeseries = load_data(SIMPLE_FIB_X_TIMESERIES_DATA_FILEPATH)
        df_x_static = load_data(FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING) # we wont use this but its nec for get_training_and_test_data
        df_y_timeseries = load_data(SIMPLE_FIB_Y_TARGET_VARIABLE_FILEPATH)

    else:
        df_x_timeseries = load_data(FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING)
        df_x_static = load_data(FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING)
        df_y_timeseries = load_data(FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING)

    train_data, test_data, val_data = get_training_and_test_data(df_x_timeseries, df_x_static, df_y_timeseries, return_padded_tensors=False, return_val_data=True)

if __name__=="__main__":
    main()
