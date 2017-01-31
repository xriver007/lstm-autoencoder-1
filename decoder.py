from keras.models import Model
from keras import layers
from base import EncoderDecoderBase


class LSTMDecoder(EncoderDecoderBase):
    def build(self):
        # unpack hyperparameters
        (time_size, word2vec_size, 
         lstm_hiddens, lstm_dense, dropout_rate)  = self._unpack_config()

        # lstm decoding layers
        #
        # note that we build two separate tensors: one for the autoencoder and 
        # the other for the decoder.
        dc_input = layers.Input(shape=(lstm_hiddens[len(lstm_hiddens) - 1],))
        d = self.input
        dc = dc_input
        for i, h in enumerate(reversed(lstm_hiddens)):
            first = i == 0

            if lstm_dense:
                dense_layer = (
                    layers.Dense(h) if first else
                    layers.TimeDistributed(layers.Dense(h))
                )
                d = dense_layer(d)
                dc = dense_layer(dc)

                bn_layer = layers.BatchNormalization()
                d = bn_layer(d)
                dc = bn_layer(dc)

                relu_layer = layers.Activation('relu')
                d = relu_layer(d)
                dc = relu_layer(dc)

                dropout_layer = layers.Dropout(dropout_rate)
                d = dropout_layer(d)
                dc = dropout_layer(dc)

            if first:
                repeat_layer = layers.RepeatVector(time_size)
                d = repeat_layer(d)
                dc = repeat_layer(dc)

            lstm_layer = layers.Bidirectional(layers.LSTM(
                h, input_length=time_size, return_sequences=True
            ))
            d = lstm_layer(d)
            dc = lstm_layer(dc)

        # classification layer
        distribution_layer = layers.TimeDistributed(layers.Dense(
            self.dimension, activation='softmax'
        ))
        d = distribution_layer(d)
        dc = distribution_layer(dc)

        # return the decoded tensor and the decoding model
        return d, Model(input=dc_input, output=dc)
