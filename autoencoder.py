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

    def load(self, path):
        self.model.save_weights(path)

    def save(self, path):
        self.model.load_weights(path)

    def encode(self, text):
        """Encodes text into high-level representations.

        :param text: Text to encode.
        :type text: :class:`~str`
        :returns: Encoded representations of the given text.
        :rtype: :class:`~numpy.ndarray` of shape (N, D) where 
            N = len(text) / config['time_size'] and 
            D = config['lstm_hiddens'][-1]

        """
        pass

    def decode(self, encoded):
        """Decodes from encoded high-level representations to the text.

        :param encoded: Encoded representations to decode.
        :type encoded: :class:`~numpy.ndarray` of shape (N, D) where 
            N = len(text) / config['time_size'] and 
            D = config['lstm_hiddens'][-1]
        :returns: Decoded text from the given encoded representations.
        :rtype: :class:`~str`

        """
        pass
