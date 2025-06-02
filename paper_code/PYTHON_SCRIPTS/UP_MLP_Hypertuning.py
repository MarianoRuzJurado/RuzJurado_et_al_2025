#!/usr/bin/env python3

import argparse, sys
from UP_utils import *
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras_tuner import Hyperband, Objective
from UP_models_hypertuning import MLP_Model_builder
from UP_preprocess import MultiLabelBin

# Define the file path for your log file
log_file_path = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/MLP_240620_3_500ep.log'

def parse_args():
    # Reading in the Commandline arguments
    argParser = argparse.ArgumentParser(description='Hypertuning MLP Script')
    argParser.add_argument('-mtp', '--MLP_trainpath', type=str, required=True, help='PATH to encoded Training Data NPY-File')
    argParser.add_argument('-tl', '--trainlabels', type=str, required=True, help='PATH to Training Label CSV-File')
    argParser.add_argument('-mvp', '--MLP_valpath', type=str, required=True, help='PATH to encoded Validation Data NPY-File')
    argParser.add_argument('-vl', '--vallabels', type=str, required=True, help='PATH to Validation Label CSV-File')
    argParser.add_argument('-of', '--outputFolder', type=str, required=False, help='PATH to output folder',
                           default=os.getcwd())
    argParser.add_argument('-bz', '--batchsize', type=int, required=False, help='Define the Batchsize as integer',
                           default=10240)
    argParser.add_argument('-s', '--seed', type=int, required=False,
                           help='Define the Seed for deterministic environment, default Value 1234', default=1234)
    argParser.add_argument('-mn', '--min_value_list', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-mx', '--max_value_list', nargs='+', type=int, required=True, default=None)
    argParser.add_argument('-sp', '--step_list', nargs='+', type=int, required=True, default=None)

    return argParser.parse_args()

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    ae_trainpath = args.AE_trainpath
    ae_valpath = args.AE_valpath

    # parameters
    min_value_list = args.min_value_list
    max_value_list = args.max_value_list
    step_list = args.step_list

    # Do the multi-label-hot encoding
    train_labels_bin = MultiLabelBin(args.trainlabels)
    val_labels_bin = MultiLabelBin(args.vallabels)

    ae_features_train = np.load(ae_trainpath)
    ae_features_val = np.load(ae_valpath)
    early_stop = EarlyStopping(monitor='val_macro_f1_loss', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_macro_f1_loss', factor=0.1, patience=25, min_lr=1e-7)  # LR reduction ADAM callback

    # Redirect stdout to the log file
    sys.stdout = open(log_file_path, 'w')
    print(min_value_list)
    print(max_value_list)
    print(step_list)
    mlp_instance = MLP_Model_builder(input_dim=ae_features_train.shape[1],
                                             num_layers=5,
                                             min_value_list=min_value_list,
                                             max_value_list=max_value_list,
                                             step_list=step_list,
                                             output_dim=13)

    tuner = Hyperband(mlp_instance,
                      objective=Objective('val_macro_f1_loss', direction='min'),
                      max_epochs=100,
                      factor=3,
                      directory=args.outputFolder,
                      project_name='mlp_hypertuning')

    tuner.search(ae_features_train, train_labels_bin,
                 epochs=100,
                 validation_data=(ae_features_val, val_labels_bin),
                 batch_size=10240,
                 callbacks=[reduce_lr, early_stop])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


    print(f"""
    The hyperparameter search is complete. The optimal number of units in the 2 layers of the MLP are {best_hps.get('unit_mlp0')} {best_hps.get('unit_mlp1')} {best_hps.get('unit_mlp2')} {best_hps.get('unit_mlp3')} {best_hps.get('unit_mlp4')}.
    """)

    # Restore stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # Reset stdout to its default value
