import abc


class EncoderDecoderBase(object, metaclass=abc.ABCMeta):
    def __init__(self, input, dimension, config=None):
        self.input = input
        self.dimension = dimension
        self._config = config or {}
        self._tensor, self._model = self.build()

    @property
    def tensor(self):
        return self._tensor

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError

    def run(self):
        return self.model.predict(X)

    def _unpack_config(self):
        return (
            self._config['time_size'],
            self._config['word2vec_size'],
            self._config['lstm_hiddens'], 
            self._config ['lstm_dense'],
            self._config['dropout_rate'],
        )
