import tensorflow as tf
from keras import layers, models
import keras.backend as K
import sys

class AttentionLayer(layers.Layer):
    def __init__(self, return_attention_weights=False, return_sequences=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.return_attention_weights = return_attention_weights
        self.return_sequences = return_sequences

    def build(self, input_shape: list):
        # input_shape will be a list of shapes: [query_shape, key_shape, value_shape]
        query_shape, key_shape, value_shape = input_shape
        
        initializer_1 = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05)
        initializer_3 = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05)
        initializer_5 = tf.keras.initializers.RandomNormal(mean=0., stddev=0.05)

        self.W_q = self.add_weight(name='attention_weight_q',
                                   shape=(query_shape[-1], query_shape[-1]),
                                   initializer=initializer_1,
                                   trainable=True)
        
        self.b_q = self.add_weight(name='attention_bias_q',
                                      shape=(query_shape[-1],),
                                      initializer='zeros',
                                      trainable=True)

        self.W_k = self.add_weight(name='attention_weight_k',
                                   shape=(key_shape[-1], key_shape[-1]),
                                   initializer=initializer_3,
                                   trainable=True)
        
        self.b_k = self.add_weight(name='attention_bias_k',
                                      shape=(key_shape[-1],),
                                      initializer='zeros',
                                      trainable=True)

        self.W_v = self.add_weight(name='attention_weight_v',
                                   shape=(value_shape[-1], value_shape[-1]),
                                   initializer=initializer_5,
                                   trainable=True)

        self.b_v = self.add_weight(name='attention_bias_v',
                                        shape=(value_shape[-1],),
                                        initializer='zeros',
                                        trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Now properly initialized

    def call(self, inputs):
        '''
        Attention mechanism for static and temporal data.
        The static data (output from the NN, Q) is used to provide context to the timerseies data (K, V).
        Attention is calculated by: 
            softmax(Q.K^T) * V 
        where
            . is the dot product.
            * is the element-wise multiplication.
            softmax(Q.K^T) are the attention weights.

        Q, K and V each have a learnable weight matrix and bias.

        In order to calculate the softmax weight, we had to reduce the dimensions of Q and K to 2D (squeeze)
        and then expand the dimensions back to 3D after the softmax calculation.

        Parameters:
            inputs: list of the Q, K and V matrices.

        Returns:
            The attention output and the attention weights (if return_attention_weights=True).

        '''

        query, key, value = inputs
        num_dimensions = key.shape[2]
        weight_q = weight_k = weight_v = None

        weight_q = K.dot(query, self.W_q) + self.b_q
        weight_k = K.dot(K.squeeze(key, axis=0), self.W_k) + self.b_k
        weight_v = K.dot(K.squeeze(value, axis=0), self.W_v) + self.b_v

        attention_scores = tf.matmul(weight_q, weight_k, transpose_b=True)
        attention_scores = K.tanh(attention_scores)
        
        squeeze = False
        if attention_scores.shape[0] == 1:
            attention_scores = K.squeeze(attention_scores, axis=0)
            squeeze = True
        elif attention_scores.shape[-1] == 1:
            attention_scores = K.squeeze(attention_scores, axis=-1)
            squeeze = True
        else:
            print('No squeeze! ', attention_scores.shape)

        attention_weights = tf.nn.softmax(attention_scores)

        if squeeze:
            attention_weights = K.expand_dims(attention_weights, axis=0)
            weight_v = K.expand_dims(weight_v, axis=0)

        repeated_attention_weights = layers.RepeatVector(num_dimensions)(attention_weights)
        repeated_attention_weights = layers.Permute([2,1])(repeated_attention_weights) 
        attention_output = layers.Multiply()([weight_v, repeated_attention_weights])

        # sum over the timesteps
        attention_output = K.sum(attention_output, axis=2, keepdims=True)

        if self.return_attention_weights:
            return attention_output, attention_weights

        return attention_output
