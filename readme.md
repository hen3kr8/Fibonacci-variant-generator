# Forecasting Fibonacci with attention 

This model forecasts the next 10 elements of a variant of the fibonacci sequence by combining static and timeseries data using an attention mechanism. 

![A diagram of the model architecture](images/att_architecture_diagram.png)




## How to Run

1. Install the necessary dependencies by running the following command:
    ```
    pip install requirements.txt
    ```

The data has already been created and scaled. All that is needed is to run the model.


2. To train and test the model run the following command:
    ```
    python -m BuildModels.create_fib_attention_model
    ```


## Context

The original Fibonacci sequence is
a sequence of integers (elements) where each integer is the sum of the
preceding two values. The variant we have created includes characteristics
which describe the relationship between the input and output sequence,
such as whether the sequence is in reverse, has been multiplied by a
scalar, contains noise and the number of elements left out between the two
sequences

The objective of the dataset is to use the input Fibonacci sequence
(X_sequence) as well as characteristics of the sequence (X_static) to predict the
next 10 elements in the sequence (Y_sequence).
The static data in this problem is critical to accurately predict the
next sequence, which makes it a suitable problem for testing whether the
architectures we are using can indeed learn the correlation between the static
and time series data

The script used to generate the data can be found at [FibData/create_fib_data.py](FibData/create_fib_data.py).

The model trains on static (X_Static) and timeseries (X_sequence) data to forecast the following sequence (Y_sequence). 


## Table 1: Variables used in Fibonacci dataset

| Variable Name | Description | Data Type | Range | Dataset |
|---------------|-------------|-----------|-------|---------|
| Fib_1         | 1st element in sequence | Discrete | [1, 49] | Static |
| Fib_2         | 2nd element in sequence | Discrete | [1, 49] | Static |
| Gap_XY        | Number of elements between the last element of Xsequence and 1st element of Ysequence | Discrete | [0, 9] | Static |
| Noise present | Whether the sequence is noisy. 10% of the sequences contain noise | Binary | [0, 1] | Static |
| Noise mean    | Noise is sampled from a normal distribution with mean = 1 and varying standard deviation. | Discrete | [0, 1] | Static |
| Noise std     | Standard deviation used to sample the noise. | Discrete | [0, 4] | Static |
| Reversed      | Whether the sequence is in decreasing order or not. If reversed, Fib_1, Fib_2 are the last elements in Ysequence. Else first elements in Xsequence. 50% of sequences are reversed. | Binary | [0, 1] | Static |
| Multiplier    | Scalar with which sequence has been multiplied. | Discrete | [1, 4] | Static |
| X_sequence     | Elements of Fibonacci sequence used as input. All sequences length 10. |  | | Timeseries |
| Y_sequence     | Elements of the Fibonacci sequence to predict. All sequences length 10. |  | | Target|




## Other models:

We compared our architecture to hybrid and ensemble architectures (See diagrams below), and found that the attention architecture achieves similar performance. 


![A diagram of the hybrid architecture](images/hybrid_diagram.png)
Figure 1: A diagram of the hybrid architecture

![A diagram of the ensemble/meta architecture](images/meta_arch.png)
Figure 2: A diagram of the ensemble/meta architecture


## Results:

![A boxplot of the MSE of the different architectures](images/Fib_box_plot_excl_meta_no_seq.png)
Figure 3: A boxplot of the MSE of the different architectures



## Attention weights

Below we have visualised the attention weights. We see elements 8 and 9 have the highest weight. This means the model deems them as most important when predicting the following sequence. This does reflect a structural dependency in the fibonacci sequence, where the current element is determined by the sum of the previous 2 elements.
This could suggest that attention mechansisms can be used to reveal or validate structural dependencies within data. 

![The attention weights for the Fib architecture](images/fib_att_weights.png)
Figure 3: The attention weights for the Fib architecture



## Conclusion:

Because the attention mechanism uses a dot product instead of concatenating the static and timeseries data, the dimensionality of the timeseries data fed into the  stays the same, meaning the model has less weights to learn, making it less prone to overfitting. 

For a visual of what I mean, see image [every_step_concatenation.png](images/every_step_concatenation.png).

An added bonus is that attention weights can be visualized, which may or may not reveal structural dependencies within the data. 


