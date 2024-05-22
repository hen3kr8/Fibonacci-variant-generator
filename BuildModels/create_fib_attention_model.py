# attention model for fibnacci sequence data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
from BuildModels import create_tensors
from BuildModels import fib_helper_functions
from BuildModels.attention_layer import AttentionLayer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from tensorflow.python.ops.numpy_ops import np_config
import seaborn as sns
np_config.enable_numpy_behavior()

FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING = 'ScaledData/all_X_timeseries_fib_after_scaling.xlsx'
FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING = 'ScaledData/all_Y_timeseries_fib_after_scaling.xlsx'
FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING = 'ScaledData/all_X_static_fib_after_scaling.xlsx'
X_TIMESERIES_SCALER_PATH = 'ScaledData/Scalers/x_timeseries_scaler.save'
Y_TIMESERIES_SCALER_PATH = 'ScaledData/Scalers/y_timeseries_scaler.save'

BATCH_SIZE = 1
RETURN_SEQUENCES = True
SEQUENCE_LENGTH = 10
tf.random.set_seed(1337)

def load_all_data():
    df_x_timeseries = pd.read_excel(FIB_X_TIMESERIES_DATA_FILEPATH_AFTER_SCALING)
    df_x_static = pd.read_excel(FIB_X_STATIC_DATA_FILEPATH_AFTER_SCALING)
    df_y_timeseries = pd.read_excel(FIB_Y_TARGET_VARIABLE_FILEPATH_AFTER_SCALING)
    train_data, test_data, val_data = create_tensors.get_training_and_test_data(df_x_timeseries, df_x_static, df_y_timeseries, return_padded_tensors=False, return_val_data=True)

    return train_data, test_data, val_data

def create_attention_decoder_model(
    X_train_static:tf.Tensor, 
    X_train_timeseries:tf.Tensor, 
    y_train_timeseries:tf.Tensor, 
    X_val_static:tf.Tensor, 
    X_val_timeseries:tf.Tensor, 
    y_val_timeseries:tf.Tensor, 
    config: dict
    ) -> Tuple[tf.keras.Model, tf.keras.Model, pd.DataFrame]:

    """
    Creates an attention-based decoder model combining static and 
    timeseries data for time series forecasting.

    The model works as follows:
    1. Static data is fed through a NN
    2. Timeseries data through an LSTM.
    3. NN and LSTM output is then fed through an attention layer, which 
    calculates the attention weights by taking the dot product of these two.
    The attention weights are multiplied by the timeseries data
    4. The result is fed through an LSTM decoder and a final dense layer.


    Args:
        X_train_static (tf.Tensor): Input tensor for static features in the training set.
        X_train_timeseries (tf.Tensor): Input tensor for time series features in the training set.
        y_train_timeseries (tf.Tensor): Target tensor for time series in the training set.
        X_val_static (tf.Tensor): Input tensor for static features in the validation set.
        X_val_timeseries (tf.Tensor): Input tensor for time series features in the validation set.
        y_val_timeseries (tf.Tensor): Target tensor for time series in the validation set.
        config (dict): Configuration dictionary containing hyperparameters and settings.

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model, pd.DataFrame]: A tuple containing the main model, attention model,
        and a DataFrame with training history.

    Examples:
        model, attention_model, results = create_attention_decoder_model(
            X_train_static, X_train_timeseries, y_train_timeseries,
            X_val_static, X_val_timeseries, y_val_timeseries,
            config
        )
    """
    
    minval = -0.2 
    maxval = 0.2  
    bias_initializer = tf.keras.initializers.RandomUniform(minval=minval, maxval=maxval, seed=1337)
    kernel_initializer_1 = tf.keras.initializers.RandomUniform(minval=minval, maxval=maxval)
    kernel_initializer_4 = tf.keras.initializers.RandomUniform(minval=minval, maxval=maxval)
    kernel_initializer_rnn = tf.keras.initializers.GlorotUniform() 
    bias_initializer_1 = tf.keras.initializers.RandomUniform(minval=minval, maxval=maxval)
    bias_initializer_4 = tf.keras.initializers.RandomUniform(minval=minval, maxval=maxval)
    bias_initializer_5 = tf.keras.initializers.Zeros()

    static_input_shape = (X_train_static.shape[1], )  # Assuming X_train_static is a 2D array
    static_model_input = tf.keras.layers.Input(
        shape=static_input_shape, 
        batch_size=BATCH_SIZE
        )

    # Static model
    x = tf.keras.layers.Dense(
        config["num_units"],
        activation=config['activation'],
        kernel_initializer=kernel_initializer_1,
        bias_initializer=bias_initializer_1)(static_model_input)
    x = tf.keras.layers.Dropout(0.2)(x)

    for _ in range(config["num_layers"]):
        kernel_initializer_2 = tf.keras.initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=np.random.randint(1, 1000)
        )
        bias_initializer_2 = tf.keras.initializers.RandomUniform(
            minval=minval, maxval=maxval, seed=np.random.randint(1, 1000)
        )
        x = tf.keras.layers.Dense(
            config["num_units"], 
            activation=config['activation'], 
            kernel_initializer=kernel_initializer_2, 
            bias_initializer=bias_initializer_2
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    static_output = tf.keras.layers.Dense(
        config["num_units"], 
        activation=config['activation'], 
        kernel_initializer=bias_initializer
    )(x)
    
    # Timeseries model
    timeseries_input_shape = (X_train_timeseries.shape[1], X_train_timeseries.shape[2])  # Assuming X_train_timeseries is a 3D array
    timeseries_model_input = tf.keras.layers.Input(
        shape=timeseries_input_shape, 
        batch_size=BATCH_SIZE
        )
    
    rnn_outputs, forward_rnn_h, forward_rnn_c, back_rnn_h, back_rnn_c = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=config["num_units"]//2,
            return_sequences=RETURN_SEQUENCES,
            activation=config['activation'],
            kernel_initializer=kernel_initializer_rnn,
            bias_initializer=bias_initializer_5, 
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4), 
            return_state=True
        ),
    )(timeseries_model_input)

    state_h = tf.keras.layers.Concatenate()([forward_rnn_h, back_rnn_h])
    state_c = tf.keras.layers.Concatenate()([forward_rnn_c, back_rnn_c])

    # output of static and timeseries model combined in attention layer
    attention_layer = AttentionLayer(
        return_attention_weights=True, 
        return_sequences=RETURN_SEQUENCES, 
        name="attention_layer"
        )    

    attention_output, _ = attention_layer([
                                            static_output,
                                            rnn_outputs,
                                            rnn_outputs
                                        ])

    # 2nd LSTM layer, to learn correlation in the attention output
    decoder_lstm = tf.keras.layers.LSTM(
        config["num_units"],
        return_sequences=True, 
        activation=config['activation'], 
        kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        )
    
    decoder_outputs = decoder_lstm(attention_output, initial_state=[state_h, state_c])
    
    # Final output layer with single neuron, to reduce dimension to 1
    final_output = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=kernel_initializer_4,
        bias_initializer=bias_initializer_4
    )(decoder_outputs)
    
    model = tf.keras.Model(
        inputs=[static_model_input, timeseries_model_input],
        outputs=[final_output]
    )
    
    attention_model = tf.keras.Model(
        inputs=model.input, 
        outputs=model.get_layer("attention_layer").output
    )
    
    adam_optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(loss=config['loss'], optimizer=adam_optimizer, metrics=config['loss'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error',  
        patience=config['patience'],         
        restore_best_weights=config['Restore_best_weights'], 
        start_from_epoch=config['start_from_epoch'] 
    )
    history = model.fit(
        [X_train_static, X_train_timeseries], 
        [y_train_timeseries],
        epochs=config['num_epochs'], 
        verbose=1,
        batch_size=BATCH_SIZE,
        validation_data=([X_val_static, X_val_timeseries], [y_val_timeseries]),
        callbacks= [early_stopping] if config['early_stopping'] else None,
        )
    results = pd.DataFrame(history.history)
    
    print(model.summary())
    return model, attention_model, results


def test_model_with_attention(
        attention_model:tf.keras.Model, 
        X_test_static:tf.Tensor, 
        X_test_timeseries:tf.Tensor, 
        filename=None
    ):
    # Predict with your model to get the attention weights per test sample, then plot weights
    # Make sure to set return_attention_weights=True if using the subclassed model approach
    
    _, attention_weights = attention_model.predict(
        [X_test_static, X_test_timeseries], 
        verbose=1, 
        batch_size=BATCH_SIZE
    )
    sample_index = 0
    sample_attn_weights = attention_weights[sample_index]

    average_attention_weights = np.mean(attention_weights, axis=0)

    x = [i for i in range(len(average_attention_weights))]
    plt.bar(x, average_attention_weights)
    plt.xlabel('Time steps')
    plt.ylabel('weight size')
    plt.xticks(x)
    plt.title('Fib data - Average Attention weights for single sample')
    plt.grid(True)
    plt.savefig(filename+'average_attention_weights_single_sample.png')
    plt.close()

    # Assuming attn_weights are in the shape you expect, e.g., (num_time_steps, num_static_features)
    try:
        plt.matshow(attention_weights)
        plt.xlabel('Time steps')
        plt.ylabel('Sample')
        plt.title('Fib data - attention weights')
        plt.savefig(filename+'attention_weights_all_1.png')
        # plt.show()
        plt.close()
    except Exception as e:
        print(e)
    
    x = [i for i in range(len(sample_attn_weights))]
    plt.bar(x, sample_attn_weights)
    plt.xlabel('Time steps')
    plt.ylabel('weight size')
    plt.xticks(x)
    plt.title('Fib data - Attention weights for single sample')
    plt.grid(True)
    plt.savefig(filename+'attention_weights_single_sample.png')
    plt.close()

    sample_attn_weights = np.squeeze(sample_attn_weights)
    sns.heatmap(attention_weights, cmap='viridis')
    plt.xlabel('Time steps')
    plt.ylabel('Sample')
    plt.title('Fib data - Attention weights')
    # plt.savefig(filename+'attention_weights_all_2.png')
    plt.show()
    plt.close()


def run_attention_decoder_experiment():
    """
    Runs an experiment using the attention decoder model for the Fibonacci 
    variant generator.

    This function loads the training, testing, and validation data, and then 
    iterates over a set of hyperparameters to train and evaluate the attention 
    decoder model. 
    The hyperparameters include: 
        - learning rate 
        - number of neural network units
        - number of neural network layers. 
    
    Only a single hyperparameter is tuned at a time.
    
    Set the 'param_to_tune' variable to the hyperparameter you want to tune 
    (ex. 'learning_rate'). Currently it is set to 'final_model', using the (i 
    guess you could say hardcoded) hyperparameters, as these parameters have 
    found to produce the best results.
    
    To run multiple iterations, set the 'num_iterations' variable to the number


    If you wish to visualise the attention weights, set the 
    'visualise_attention_weights' variable to True.

    The function saves the trained models and writes the results
    to an Excel file.

    Returns:
        None
    """
    
    train_data, test_data, val_data = load_all_data()
    X_train_static, X_train_timeseries, y_train_timeseries = train_data
    X_test_static, X_test_timeseries, y_test_timeseries = test_data
    X_val_static, X_val_timeseries, y_val_timeseries = val_data
   
    # range in which hyperparameters are tuned
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    num_nn_units = [16, 32, 64, 256]
    num_nn_layers = [0,1,2,3,4] 
    num_iterations = 1

    param_to_tune = 'final_model'
    model_to_train = 'attention_decoder'
    learning_rate = 0.0001
    num_units = 64
    num_layers = 0

    for i in range(num_iterations):
        run = i
        model = None
        results = None
        attention_model = None
        visualise_attention_weights = False

        if param_to_tune == 'learning_rate':
            learning_rate = learning_rates[i % len(learning_rates)]

        elif param_to_tune == 'num_units':
            num_units = num_nn_units[i % len(num_nn_units)]

        elif param_to_tune == 'num_layers':
            num_layers = num_nn_layers[i % len(num_nn_layers)]

        config = {
            'learning_rate': learning_rate,
            'activation': 'relu',
            'loss': MeanSquaredError(),
            'num_layers': num_layers,
            'num_units': num_units,
            'num_rnn_units': 64,
            'early_stopping': True,
            'patience': 2,
            'start_from_epoch': 100,
            'Restore_best_weights': True,
            'num_epochs': 2
            }
        
        print('STARTING RUN ' + str(i) + ' for ' + model_to_train)

        model, attention_model, results = create_attention_decoder_model(
            X_train_static, 
            X_train_timeseries, 
            y_train_timeseries, 
            X_val_static, 
            X_val_timeseries, 
            y_val_timeseries, 
            config
        )
            
        if visualise_attention_weights:
            test_model_with_attention(
                attention_model, 
                X_test_static, 
                X_test_timeseries, 
                filename='attention_weights/attention_decoder_'+str(i)+'_'
            )

        fib_helper_functions.save_model(
            model, 
            'SerializedModels/' + model_to_train + '_'+ param_to_tune + '_' + str(i)+ '.h5'
        )
        test_acc = fib_helper_functions.test_model(
            model, 
            [X_test_static, X_test_timeseries], 
            y_test_timeseries, 
            static=True
        )

        test_acc = test_acc[1]
        fib_helper_functions.test_single_sample_static_and_timeseries(
            model, 
            [X_test_static[0], X_test_timeseries[0]], 
            y_test_timeseries[0]
        )

        results_excel = {
                'excel_file': 'Results/' + model_to_train + '_' + str(param_to_tune) + '.xlsx',
                'run': run,
                'learning_rate': config['learning_rate'],
                'epochs': len(results),
                'num_layers': config['num_layers'],
                'num_units': config['num_units'],
                'test_mse': test_acc,
                'final_training_mse': results['mean_squared_error'].iloc[-1],
                'mse': results['mean_squared_error'],
                'val_mse': results['val_mean_squared_error'],
            }
        fib_helper_functions.write_run_to_excel(results_excel)
        print('FINISHED RUN ' + str(i) + ' for ' + model_to_train)

def main():
    run_attention_decoder_experiment()


if __name__=="__main__":
    main()