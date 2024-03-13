# MIT License
# 
# Copyright (c) [2024] [Hendrik Carel Krijt]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
from typing import List
from typing import Tuple

PERCENTAGE_NOISE_IN_DATA = 0.1

def fibonacci_log(fib_series:List[int])-> List[int]:
    # log the fibonacci series
    log_fib = np.log(np.asarray(fib_series))
    print(log_fib)
    return log_fib

def fibbonacci_gap_XY(fib_1: int, fib_2: int, sequence_length=10, multiplier=1, reversed=0, gap_XY = 0, noise=0, noise_mean=0, noise_std=0) -> Tuple[List[int], List[int]]:
    ''' 
    Args:
        fib_1: first number in the fibonacci series
        fib_2: second number in the fibonacci series
        n: length of the fibonacci series
        m: scalar multiplier
        r: reverse the sequence
        gap_XY: the number of elements between the last element in X and Y the first element in Y

    Returns:
        X_fib: the X sequence
        Y_fib: the Y sequence

    '''
    fib_series = fibonacci(fib_1, fib_2, 2*sequence_length+gap_XY)

    if reversed:
        fib_series = [i*multiplier for i in fib_series[::-1]]
    else:
        fib_series = [i*multiplier for i in fib_series]

    X_fib = fib_series[:sequence_length]
    Y_fib = fib_series[sequence_length+gap_XY:2*sequence_length+gap_XY]

    if noise:
        X_fib = add_noise(X_fib, noise_mean, noise_std)
        Y_fib = add_noise(Y_fib, noise_mean, noise_std)

    return X_fib, Y_fib

def fibonacci(fib_1:int, fib_2:int, n:int):
    """
    Generate a Fibonacci sequence.

    Parameters:
        fib_1 (int): The first number in the Fibonacci sequence.
        fib_2 (int): The second number in the Fibonacci sequence.
        n (int): The length of the Fibonacci sequence.

    Returns:
        array-like: The Fibonacci sequence.
    """

    fib_series = [fib_1, fib_2]
    for i in range(2,n):
        fib_n = fib_1 + fib_2
        fib_series.append(fib_n)
        fib_1 = fib_2
        fib_2 = fib_n
    return fib_series

def add_noise(fib_seq: List[int], noise_mean: int, noise_level: int) -> List[int]:
    """
    Adds Gaussian noise to a Fibonacci sequence.

    Parameters:
    fib_seq (array-like): The Fibonacci sequence to which noise will be added.
    noise_mean (int): The mean of the Gaussian noise.
    noise_level (int): The standard deviation of the Gaussian noise.

    Returns:
    array-like: The Fibonacci sequence with added Gaussian noise.
    """

    noisy_seq = fib_seq + np.random.normal(noise_mean, noise_level, len(fib_seq))
    noisy_seq = np.round(noisy_seq)
    return noisy_seq

def create_simple_fib_data():
    '''
    THis is the original function that creates the fibonacci series, with no noise or reversal.
    The X data is the original fibonacci series, and the Y data is the the next 10 elemenets of the sequence.
    There is no static data. The RNN model should be able to learn the sequence from the X data.
    This serves as a benchmark for the the other models.
    
    '''
    sequence_length = 10
    fib_1_range = np.arange(1, 50, 1)
    fib_2_range = np.arange(1, 50, 1)
    all_X_time_series = pd.DataFrame()
    all_Y_time_series = pd.DataFrame()

    for i in range(1000):
        fib_1 = np.random.choice(fib_1_range)
        fib_2 = np.random.choice(fib_2_range)
        fib_series = fibonacci(fib_1, fib_2, 2*sequence_length)
        fib_X = fib_series[:sequence_length]
        fib_Y = fib_series[sequence_length: 2*sequence_length]

        fib_X_df = pd.DataFrame(fib_X, columns=['sequence'])
        fib_X_df['id'] = i
        fib_X_df = fib_X_df[['id', 'sequence']]
        fib_X_df['position'] = fib_X_df.groupby('id')['id'].rank(method='first')
        
        fib_Y_df = pd.DataFrame(fib_Y, columns=['sequence'])
        fib_Y_df['id'] = i
        fib_Y_df = fib_Y_df[['id', 'sequence']]
        fib_Y_df['position'] = fib_Y_df.groupby('id')['id'].rank(method='first')

        all_X_time_series = all_X_time_series.append(fib_X_df, ignore_index=True)
        all_Y_time_series = all_Y_time_series.append(fib_Y_df, ignore_index=True)

    write_to_excel(all_X_time_series, 'simple_X_timeseries_fib.xlsx')
    write_to_excel(all_Y_time_series, 'simple_Y_timeseries_fib.xlsx')

def create_fib_data():
    """
    Generate synthetic Fibonacci data.

    This function generates synthetic Fibonacci data by randomly selecting values for various parameters,
    such as the starting Fibonacci numbers, sequence length, gaps, noise presence, and other factors.
    It creates a dataset of static data and time series data for the Fibonacci sequences.
    This data is written to Excel files. There after it can be scaled, converted to tensors and used to train a model.

    Static data includes the following columns:
    - id: the unique identifier for each sequence
    - fib_1: the first number in the Fibonacci sequence
    - fib_2: the second number in the Fibonacci sequence
    - gap_XY: the number of elements between the last element in the X sequence and the first element in the Y sequence.
    - noise_present: a binary value indicating whether noise is present in the sequence
    - noise_mean: the mean of the Gaussian noise
    - noise_std: the standard deviation of the Gaussian noise
    - reversed: a binary value indicating whether the sequence is reversed
    - multiplier: a value used to multiply the sequence

    Returns:
        None
    """

    fib_1_range = np.arange(1, 50, 1)
    fib_2_range = np.arange(1, 50, 1)
    sequnece_length = 10
    sequence_gap_X_Y_range = np.arange(0, 10, 1)
    noise_present = np.array([1, 0])
    noise_mean_range = 1
    noise_std_range = np.arange(0, 5, 1)
    reversed_range = np.array([1, 0])
    multiplier_range = np.arange(1, 5, 1)

    dataset_size = 1000
    all_static_data_df = pd.DataFrame(columns=['id', 'fib_1', 'fib_2', 'gap_XY', 'noise_present', 'noise_mean', 'noise_std', 'reversed', 'multiplier'])
    all_X_time_series = pd.DataFrame()
    all_Y_time_series = pd.DataFrame()

    for i in range(dataset_size):
        fib_1 = np.random.choice(fib_1_range)
        fib_2 = np.random.choice(fib_2_range)
        gap_XY = np.random.choice(sequence_gap_X_Y_range)
        noise = np.random.choice(noise_present, p=[PERCENTAGE_NOISE_IN_DATA, 1 - PERCENTAGE_NOISE_IN_DATA])
        if noise:
            noise_mean = 1
            noise_std = np.random.choice(noise_std_range)
        else:
            noise_mean = 0
            noise_std = 0
        reversed = np.random.choice(reversed_range, p=[0.5, 0.5])
        multiplier = np.random.choice(multiplier_range)

        # create the fibonacci series
        fib_X, fib_Y = fibbonacci_gap_XY(fib_1, fib_2, sequnece_length, multiplier, reversed, gap_XY, noise, noise_mean, noise_std)

        fib_X_df = pd.DataFrame(fib_X, columns=['sequence'])
        fib_X_df['id'] = i
        fib_X_df = fib_X_df[['id', 'sequence']]
        
        fib_X_df['position'] = fib_X_df.groupby('id')['id'].rank(method='first')
        
        fib_Y_df = pd.DataFrame(fib_Y, columns=['sequence'])
        fib_Y_df['id'] = i
        fib_Y_df = fib_Y_df[['id', 'sequence']]
        fib_Y_df['position'] = fib_Y_df.groupby('id')['id'].rank(method='first')

        all_X_time_series = all_X_time_series.append(fib_X_df, ignore_index=True)
        all_Y_time_series = all_Y_time_series.append(fib_Y_df, ignore_index=True)

    write_to_excel(all_static_data_df, 'all_X_static_fib.xlsx')
    write_to_excel(all_X_time_series, 'all_X_timeseries_fib.xlsx')
    write_to_excel(all_Y_time_series, 'all_Y_timeseries_fib.xlsx')

def write_to_excel(df, file_name):
    df.to_excel(file_name, index=False)
    print('Written to ' + file_name)

def main():
    create_fib_data()
    # create_simple_fib_data()

if __name__=="__main__":
    main()
    