from keras import layers, Model
import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class DEAutoencoder(Model):
    """Current architecture of the autoencoder"""
    def __init__(self, input_dim, num_layers, num_neurons):
        super(DEAutoencoder, self).__init__()

        # Extract the last num_layers neurons for decoder
        num_neurons_decoder = num_neurons[-(num_layers-1):][::1]

        # Encoder
        self.encoder = tf.keras.Sequential()
        for i in range(num_layers):
            self.encoder.add(layers.Dense(num_neurons[i], activation='relu'))

        # Decoder
        self.decoder = tf.keras.Sequential()
        for neurons_rev in num_neurons_decoder:  # Start at reversed order, Stop at -1, Step in -1 value decrease
            self.decoder.add(layers.Dense(neurons_rev, activation='relu'))
        self.decoder.add(layers.Dense(input_dim))  # Reconstruct to old dimensions

    def call(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):

        # Extract number of neurons from encoder and decoder layers
        encoder_neurons = [layer.units for layer in self.encoder.layers if hasattr(layer, 'units')]
        decoder_neurons = [layer.units for layer in self.decoder.layers[:-1] if hasattr(layer, 'units')]

        # Combine the neurons from encoder and decoder
        combined_neurons = encoder_neurons + decoder_neurons

        config = super().get_config()
        config.update({
            'input_dim': self.decoder.layers[-1].units,  # This assumes the last layer of decoder is the input_dim
            'num_layers': len(self.encoder.layers),
            'num_neurons': combined_neurons,
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract the necessary configurations
        input_dim = config.pop('input_dim')
        num_layers = config.pop('num_layers')
        num_neurons = config.pop('num_neurons')

        return cls(input_dim=input_dim, num_layers=num_layers, num_neurons=num_neurons)

@tf.keras.saving.register_keras_serializable()
class MLP_Model(Model):
    """Current architecture of the MLP"""
    def __init__(self, input_dim, output_dim, num_layers, num_neurons):
        super(MLP_Model, self).__init__()

        # Make sure there are the same amount of provided Neuron numbers and layers
        if len(num_neurons) != num_layers:
            raise ValueError('Length of num_neurons list must be equal num_layers!')

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.output_dim = output_dim

        # Group layers into a Sequential model for simplicity
        self.main_layers = tf.keras.Sequential()
        self.main_layers.add(layers.Dense(num_neurons[0], activation='relu'))
        for i in range(1, num_layers):
            self.main_layers.add(layers.Dense(num_neurons[i], activation='relu'))
        self.main_layers.add(layers.Dense(output_dim, activation='sigmoid'))

    def call(self, x):
        return self.main_layers(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'num_neurons': self.num_neurons,
            'output_dim': self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract necessary configurations to reconstruct the model
        input_dim = config.pop('input_dim')
        num_layers = config.pop('num_layers')
        num_neurons = config.pop('num_neurons')
        output_dim = config.pop('output_dim')

        # Create a new instance using the extracted parameters
        return cls(input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, num_neurons=num_neurons)

