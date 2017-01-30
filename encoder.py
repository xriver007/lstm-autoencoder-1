from keras.models import Model
from keras import layers
from base import EncoderDecoderBase


class LSTMEncoder(EncoderDecoderBase):
    def build(self):
        # unpack hyperparameters
        (time_size, word2vec_size, 
         lstm_hiddens, lstm_dense, 
         dropout_rate)  = self._unpack_config()

        # embedding layer
        input = layers.Input(shape=(self.dimension,))
        e = layers.Embedding(self.dimension, word2vec_size)(input)

        # lstm encoding layers
        for i, h in enumerate(lstm_hiddens):
            last = i == len(lstm_hiddens) - 1

            e = layers.Bidirectional(layers.LSTM(
                h, input_length=time_size, return_sequences=(not last)
            ))(e)

            if lstm_dense:
                dense_layer = (
                    layers.Dense(h) if last else 
                    layers.TimeDistributed(layers.Dense(h))
                )
                e = dense_layer(e)
                e = layers.BatchNormalization()(e)
                e = layers.Activation('relu')(e)
                e = layers.Dropout(dropout_rate)(e)

        # return the encoding layer / model
        return e, Model(input, e)
