from keras.models import Model
from keras import layers
from base import EncoderDecoderBase


class LSTMDecoder(EncoderDecoderBase):
    def build(self):
        # unpack hyperparameters
        (time_size, word2vec_size, 
         lstm_hiddens, lstm_dense, dropout_rate)  = self._unpack_config()

        # lstm decoding layers
        d = input = layers.Input(shape=(lstm_hiddens[len(lstm_hiddens) - 1],))
        for i, h in enumerate(reversed(lstm_hiddens)):
            first = i == 0

            if lstm_dense:
                dense_layer = (
                    layers.Dense(h) if first else
                    layers.TimeDistributed(layers.Dense(h))
                )
                d = dense_layer(d)
                d = layers.BatchNormalization()(d)
                d = layers.Activation('relu')(d)
                d = layers.Dropout(dropout_rate)(d)

            if first:
                d = layers.RepeatVector(time_size)(d)

            d = layers.Bidirectional(layers.LSTM(
                h, input_length=time_size, return_sequences=True
            ))(d)

        # classification layer
        d = layers.TimeDistributed(layers.Dense(
            self.dimension, activation='softmax'
        ))(d)

        # return the decoding layer / model
        return d, Model(input, d)
