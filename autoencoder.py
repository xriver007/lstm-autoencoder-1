from keras import layers
from keras.optimizers import Adam
from keras.models import Model
from encoder import LSTMEncoder
from decoder import LSTMDecoder


class Autoencoder(object):
    def __init__(self, dimension, config=None):
        time_size = config['time_size']
        # define the input
        input = layers.Input(shape=(time_size,))

        # define the encoder
        self._encoder = LSTMEncoder(input, dimension, config=config)
        encoded = self.encoder.tensor

        # define the decoder
        self._decoder = LSTMDecoder(encoded, dimension, config=config)
        decoded = self.decoder.tensor

        # define and compile autoencoder
        self._model = Model(input=input, output=decoded)
        self._model.compile(
            optimizer=Adam(lr=config['lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        self._model.summary()

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def model(self):
        return self._model
