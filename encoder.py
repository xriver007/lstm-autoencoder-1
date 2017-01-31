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
        e = layers.Embedding(
            self.dimension, 
            word2vec_size,
            input_length=time_size,
        )(self.input)

        # lstm encoding layers
        for i, h in enumerate(lstm_hiddens):
            last = i == len(lstm_hiddens) - 1

            e = layers.Bidirectional(layers.LSTM(
                h // 2, input_length=time_size, return_sequences=(not last)
            ))(e)

            if lstm_dense:
                dense_layer = (
                    layers.TimeDistributed(layers.Dense(h)) if not last else
                    layers.Dense(h)
                )
                e = dense_layer(e)
                e = layers.BatchNormalization()(e)
                e = layers.Activation('relu')(e)
                e = layers.Dropout(dropout_rate)(e)


        # return the encoded tensor and the encoding model
        return e, Model(input=self.input, output=e)
