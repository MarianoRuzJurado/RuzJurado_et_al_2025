from keras import layers
from tensorflow import keras
import tensorflow as tf
from utils import *
from keras_tuner import HyperModel
from keras_tuner import HyperParameters

class DEAutoencoder_builder(HyperModel):
    """Current hyper tuning architecture of the autoencoder"""
    def __init__(self, input_dim, num_layers, min_value_list_encoder, min_value_list_decoder, max_value_list_encoder,max_value_list_decoder, step_list_encoder, step_list_decoder):
        # super(DEAutoencoder_builder, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.min_value_list_encoder = min_value_list_encoder
        self.min_value_list_decoder = min_value_list_decoder
        self.max_value_list_encoder = max_value_list_encoder
        self.max_value_list_decoder = max_value_list_decoder
        self.step_list_encoder = step_list_encoder
        self.step_list_decoder = step_list_decoder

    def build(self, hp):
        # Create Tuners for layers
        hp_unit_list_encoder = []
        hp_unit_list_decoder = []
        for i in range(self.num_layers):
            hp_unit_list_encoder.append(hp.Int(f'unit_enc{i}',
                                                 min_value=self.min_value_list_encoder[i],
                                                 max_value=self.max_value_list_encoder[i],
                                                 step= self.step_list_encoder[i]))
        for i in range(self.num_layers-1):
            hp_unit_list_decoder.append(hp.Int(f'unit_dec{i}',
                                                 min_value=self.min_value_list_decoder[i],
                                                 max_value=self.max_value_list_decoder[i],
                                                 step= self.step_list_decoder[i]))

        # Encoder
        self.encoder = tf.keras.Sequential()
        for i in range(self.num_layers):
            self.encoder.add(layers.Dense(hp_unit_list_encoder[i], activation='relu'))

        # Decoder
        self.decoder = tf.keras.Sequential()
        for i in range(self.num_layers - 2, -1, -1):  # Start at reversed order, Stop at -1, Step in -1 value decrease
            self.decoder.add(layers.Dense(hp_unit_list_decoder[i], activation='relu'))
        self.decoder.add(layers.Dense(self.input_dim))  # Reconstruct to old dimensions

        # Define the model
        inputs = keras.Input(shape=(self.input_dim,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        autoencoder = keras.Model(inputs, decoded)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mse')

        return autoencoder


class MLP_Model_builder(HyperModel):
    """Current architecture of the hyper tuning MLP"""
    def __init__(self, input_dim, output_dim, num_layers, min_value_list, max_value_list, step_list):
        #super(MLP_Model_builder, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.min_value_list = min_value_list
        self.max_value_list = max_value_list
        self.step_list = step_list

    def build(self, hp):
        # Create Tuners for layers
        hp_unit_list = []
        for i in range(self.num_layers):
            hp_unit_list.append(hp.Int(f'unit_mlp{i}',
                                               min_value=self.min_value_list[i],
                                               max_value=self.max_value_list[i],
                                               step=self.step_list[i]))

        self.main_layers = tf.keras.Sequential()
        for i in range(self.num_layers):
            self.main_layers.add(layers.Dense(hp_unit_list[i], activation='relu'))
        self.main_layers.add(layers.Dense(self.output_dim, activation='sigmoid'))

        # Define the model
        inputs = keras.Input(shape=(self.input_dim,))
        outputs = self.main_layers(inputs)
        mlp_model = keras.Model(inputs, outputs)

        # Compile the model
        mlp_model.compile(optimizer='adam', loss=macro_f1_loss, metrics=[macro_f1_loss])

        return mlp_model
