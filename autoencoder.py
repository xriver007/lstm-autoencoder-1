from keras import layers
from keras.optimizers import Adam
from keras.models import Model
from encoder import LSTMEncoder
from decoder import LSTMDecoder


class Autoencoder(object):
    def __init__(self, dimension, config=None):
        # define input, encoder and decoder layer
        self._input_layer = layers.Input(shape=(dimension,))
        self._encoder = LSTMEncoder(dimension, config=config)
        self._decoder = LSTMDecoder(dimension, config=config)

        # connect encoder and decoder layers
        import pdb
        pdb.set_trace()
        encoded = self.encoder.layer(self._input_layer)
        decoded = self.decoder.layer(encoded)

        # define and compile autoencoder
        self._model = Model(self._input_layer, decoded)
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
