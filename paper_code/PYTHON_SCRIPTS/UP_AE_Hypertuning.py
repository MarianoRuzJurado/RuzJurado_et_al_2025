#!/usr/bin/env python3

import argparse, sys
from utils import *
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras_tuner import Hyperband
from models_hypertuning import DEAutoencoder_builder

# Define the file path for your log file
log_file_path = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning_human_model/local_run/240705.log'

def parse_args():
    # Reading in the Commandline arguments
    argParser = argparse.ArgumentParser(description='Hypertuning Autoencoder Script')
    argParser.add_argument('-tp', '--trainpath', type=str, required=True, help='PATH to Training Data CSV-File')
    argParser.add_argument('-vp', '--valpath', type=str, required=True, help='PATH to Validation Data CSV-File')
    argParser.add_argument('-of', '--outputFolder', type=str, required=False, help='PATH to output folder',
                           default=os.getcwd())
    argParser.add_argument('-bz', '--batchsize', type=int, required=False, help='Define the Batchsize as integer',
                           default=10240)
    argParser.add_argument('-s', '--seed', type=int, required=False,
                           help='Define the Seed for deterministic environment, default Value 1234', default=1234)
    argParser.add_argument('-mne', '--min_value_list_encoder', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-mxe', '--max_value_list_encoder', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-spe', '--step_list_encoder', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-mnd', '--min_value_list_decoder', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-mxd', '--max_value_list_decoder', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-spd', '--step_list_decoder', nargs='+', type=int, required=True, default=None)
    return argParser.parse_args()
#5000_5300_2200_2500_300_400_50_50_50
def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    trainpath = args.trainpath
    valpath = args.valpath
    # for encoder
    min_value_list_encoder = args.min_value_list_encoder
    max_value_list_encoder = args.max_value_list_encoder
    step_list_encoder = args.step_list_encoder
    # for decoder
    min_value_list_decoder = args.min_value_list_decoder
    max_value_list_decoder = args.max_value_list_decoder
    step_list_decoder = args.step_list_decoder

    dataset_train = pd.read_csv(trainpath, skiprows=0, header=0, index_col=0, low_memory=False)
    dataset_val = pd.read_csv(valpath, skiprows=0, header=0, index_col=0, low_memory=False)
    early_stop = EarlyStopping(monitor='loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=1e-7)

    # Redirect stdout to the log file
    sys.stdout = open(log_file_path, 'w')
    print("Min neurons encoder:", min_value_list_encoder)
    print("Max neurons encoder:", max_value_list_encoder)
    print("Steps encoder:", step_list_encoder)

    print("Min neurons decoder:", min_value_list_decoder)
    print("Max neurons decoder:", max_value_list_decoder)
    print("Steps decoder:", step_list_decoder)

    autoencoder_instance = DEAutoencoder_builder(input_dim=dataset_train.shape[1],
                                                 num_layers=3,
                                                 min_value_list_encoder=min_value_list_encoder,
                                                 max_value_list_encoder=max_value_list_encoder,
                                                 step_list_encoder=step_list_encoder,
                                                 min_value_list_decoder=min_value_list_decoder,
                                                 max_value_list_decoder=max_value_list_decoder,
                                                 step_list_decoder=step_list_decoder)

    tuner = Hyperband(autoencoder_instance,
                      objective='loss',
                      max_epochs=100,
                      factor=3,
                      directory=args.outputFolder,
                      project_name="autoencoder_human_hypertuning")

    tuner.search(dataset_train, dataset_train,
                 epochs=100,
                 validation_data=(dataset_val, dataset_val),
                 batch_size=10240,
                 callbacks=[reduce_lr, early_stop])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the three layers of the encoder are {best_hps.get('unit_enc0')} {best_hps.get('unit_enc1')} {best_hps.get('unit_enc2')}.
    """)

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the two layers of the decoder are {best_hps.get('unit_dec0')} {best_hps.get('unit_dec1')}.
    """)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to its default value

if __name__ == "__main__":
    main()
