#!/usr/bin/env python3

import argparse, pickle
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from preprocess import MultiLabelBin, createDatasets
from models import DEAutoencoder, MLP_Model
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args():
    # Reading in the Commandline arguments
    argParser = argparse.ArgumentParser(description='Neural Network Training Script')
    argParser.add_argument('-tp', '--trainpath', type=str, required=True, help='PATH to Training Data CSV-File')
    argParser.add_argument('-tl', '--trainlabels', type=str, required=True, help='PATH to Training Label CSV-File')
    argParser.add_argument('-vp', '--valpath', type=str, required=True, help='PATH to Validation Data CSV-File')
    argParser.add_argument('-vl', '--vallabels', type=str, required=True, help='PATH to Validation Label CSV-File')
    argParser.add_argument('-tsp', '--testpath', type=str, required=True, help='PATH to Test Data CSV-File')
    argParser.add_argument('-tsl', '--testlabels', type=str, required=True, help='PATH to Test Label CSV-File')
    argParser.add_argument('-of', '--outputFolder', type=str, required=False, help='PATH to output folder',
                           default=os.getcwd())
    argParser.add_argument('-bz', '--batchsize', type=int, required=False, help='Define the Batchsize as integer',
                           default=10240)
    argParser.add_argument('-s', '--seed', type=int, required=False,
                           help='Define the Seed for deterministic environment, default Value 1234', default=1234)
    argParser.add_argument('-al', '--AutoLayers', type=int, required=True,
                           help='Define the number of Layers in Autoencoder, default No Autoencoder', default=None)
    argParser.add_argument('-an', '--AutoNeurons', nargs='+', type=int, required=False,
                           help='Define the number of neurons in Autoencoder Layers (e.g. 5000 2000 500), default No '
                                'Autoencoder', default=None)
    argParser.add_argument('-ml', '--MLPLayers', type=int, required=True, help='Define the number of Layers in MLP')
    argParser.add_argument('-mn', '--MLPNeurons', nargs='+', type=int, required=True,
                           help='Define the number of neurons in MLP Layers (e.g. 500 200 100)')
    return argParser.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)

    """ For Testing in python 
    class Args:
        pass
    args = Args()
    args.trainpath = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_train_set_without_Mice_AS_Neuro_p_80.csv'
    args.valpath = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_val_set_without_Mice_AS_Neuro_p_20.csv'
    args.trainlabels = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_train_set_without_Mice_AS_Neuro_labels_p_80.csv'
    args.vallabels = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_val_set_without_Mice_AS_Neuro_labels_p_20.csv'
    args.testpath = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/240519_test_set.csv'
    args.testlabels = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/240519_test_set_labels.csv'
    #####for the whole snRNA data
    args.trainpath ="/media/Storage/anndata_shap/whole_matrix_Seurat_extract/SHUFFLED_train_set_snRNA_complete_p_80.csv"
    args.valpath ="/media/Storage/anndata_shap/whole_matrix_Seurat_extract/SHUFFLED_val_set_snRNA_complete_p_20.csv"
    args.trainlabels ="/media/Storage/anndata_shap/whole_matrix_Seurat_extract/SHUFFLED_train_set_snRNA_complete_labels_p_80.csv"
    args.vallabels ="/media/Storage/anndata_shap/whole_matrix_Seurat_extract/SHUFFLED_val_set_snRNA_complete_labels_p_20.csv"
    args.testpath = "/media/Storage/anndata_shap/whole_matrix_Seurat_extract/test_set_snRNA_complete.csv"
    args.testlabels ="/media/Storage/anndata_shap/whole_matrix_Seurat_extract/test_set_snRNA_complete_labels.csv"
    ####
    args.batchsize = 10240
    args.AutoLayers = None
    args.AutoNeurons = None
    args.seed = 1234
    tf.random.set_seed(1234)
    args.MLPLayers = 3
    args.MLPNeurons = [795, 230, 105]
    args.outputFolder = '/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18' # change for hyperparamter tuning
    # Loader commands for models
    autoencoder_instance = tf.keras.models.load_model('/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_23_10_16/AUTO_5000_2000_500_Batch_10240_SEED_1234.keras', custom_objects={'DEAutoencoder': DEAutoencoder})
    mlp_model_instance = tf.keras.models.load_model('/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_23_10_16/AE_True_5000_2000_500_MLP_500_200_100_Batch_10240_SEED_1234.keras', custom_objects={'MLP_Model':MLP_Model}, compile=True)
    args.AutoLayers = 3
    args.AutoNeurons = [5000,2400,350,2200,5150]
    """

    # Set Autoencoder build if args provided
    if args.AutoLayers is None and args.AutoNeurons is None:
        build_autoencoder = False
        ae_string = '/AE_False'
    else:
        build_autoencoder = True
        ae_string = '/AE_True_' + '_'.join(map(str, args.AutoNeurons))

    # Do the multi-label-hot encoding
    train_labels_bin = MultiLabelBin(args.trainlabels)
    val_labels_bin = MultiLabelBin(args.vallabels)
    test_labels_bin = MultiLabelBin(args.testlabels)
    # Set Save string for MLP
    MLP_neurons_str = '_'.join(map(str, args.MLPNeurons))

    np.savetxt(args.outputFolder + ae_string + '_test_labels_binarized_' + 'MLP_' + MLP_neurons_str + '_Batch_' + str(
        args.batchsize) + '_SEED_' + str(args.seed) + '.csv',
               test_labels_bin, delimiter=',')

    # Compute the steps per epoch for training
    steps_per_epoch_train = np.ceil(len(train_labels_bin) / args.batchsize)
    steps_per_epoch_val = np.ceil(len(val_labels_bin) / args.batchsize)
    steps_per_epoch_test = np.ceil(len(test_labels_bin) / args.batchsize)

    # Create Tensor Datasets from generators
    dataset_train, dataset_val, dataset_test = createDatasets(args.trainpath, args.valpath, args.testpath,
                                                              train_labels_bin, val_labels_bin, test_labels_bin,
                                                              args.batchsize, build_autoencoder)

    early_stop = EarlyStopping(monitor='loss', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=25, min_lr=1e-7)  # LR reduction ADAM callback

    if build_autoencoder is True:
        autoencoder_instance = DEAutoencoder(input_dim=dataset_train.element_spec[0].shape[1],
                                             num_layers=args.AutoLayers,
                                             num_neurons=args.AutoNeurons)
        autoencoder_instance.compile(optimizer='adam', loss='mse')

        print(get_time() + "Start fitting autoencoder...")
        history_auto = autoencoder_instance.fit(dataset_train, epochs=500,
                                                validation_data=dataset_val,
                                                steps_per_epoch=steps_per_epoch_train,
                                                validation_steps=steps_per_epoch_val,
                                                validation_freq=1,
                                                callbacks=[reduce_lr, early_stop])
        history_auto.history['epoch'] = list(range(1, len(history_auto.history['loss']) + 1))

        with open(f'{args.outputFolder}/AE_Training_history_Batch_{args.batchsize}_SEED_{args.seed}.pkl','wb') as file_pi:
            pickle.dump(history_auto.history, file_pi)

        print(get_time() + "Training finished for autoencoder...")

        loss_curve(history_auto.history['epoch'],
                   history_auto.history['loss'],
                   history_auto.history['val_loss'],
                   save_dir=f'{args.outputFolder}/Metric_figures',
                   name="AE_Train_val_loss",
                   ylim=[0.1,0.5])

        plot_loss_graph(train_loss=history_auto.history['loss'],
                        val_loss=history_auto.history['val_loss'],
                        save_path=args.outputFolder)

        AE_neurons_str = '_'.join(map(str, args.AutoNeurons))
        autoencoder_instance.save(
            args.outputFolder + '/AUTO_' + AE_neurons_str + '_Batch_' + str(args.batchsize) + '_SEED_' + str(
                args.seed) + '.keras')
        ''' Load Autoencoder
        autoencoder_instance = tf.keras.models.load_model(
            args.outputFolder + '/AUTO_' + AE_neurons_str + '_Batch_' + str(args.batchsize) + '_SEED_' + str(
                args.seed) + '.keras', custom_objects={'DEAutoencoder': autoencoder_instance})
        '''
        print(get_time() + "Autoencoder Saved!")
        print(get_time() + "Predicting data with Autoencoder...")

        ae_features_train = autoencoder_instance.predict(dataset_train, steps=steps_per_epoch_train)
        ae_features_val = autoencoder_instance.predict(dataset_val, steps=steps_per_epoch_val)

        np.save(args.outputFolder + '/AE_features_train_' + AE_neurons_str + '_Batch_' + str(
            args.batchsize) + '_SEED_' + str(
            args.seed) + '.npy', ae_features_train)
        np.save(
            args.outputFolder + '/AE_features_val_' + AE_neurons_str + '_Batch_' + str(args.batchsize) + '_SEED_' + str(
                args.seed) + '.npy', ae_features_val)

        print(get_time() + "Prediction finished and saved!")

    if build_autoencoder is True:
        input_data = ae_features_train
        input_shape = ae_features_train.shape[1]
        input_val = ae_features_val
    else:
        input_data = dataset_train
        input_shape = dataset_train.element_spec[0].shape[1]
        input_val = dataset_val

    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, min_lr=1e-7)  # LR reduction ADAM callback

    mlp_model_instance = MLP_Model(input_shape, train_labels_bin.shape[1], args.MLPLayers, args.MLPNeurons)
    mlp_model_instance.compile(optimizer='adam', loss=macro_f1_loss,
                               metrics=[f1_mean, f1_species, f1_celltype, f1_disease,
                                        precision_mean, precision_species, precision_celltype, precision_disease,
                                        recall_mean, recall_species, recall_celltype, recall_disease], )

    # Training with encoded features
    print(get_time() + "Start fitting MLP...", )
    if build_autoencoder is True:
        history_mlp = mlp_model_instance.fit(input_data, train_labels_bin, epochs=500, batch_size=args.batchsize,
                                             validation_data=(input_val, val_labels_bin),
                                             callbacks=[reduce_lr, early_stop])
    else:
        history_mlp = mlp_model_instance.fit(input_data, epochs=500,
                                             validation_data=input_val,
                                             steps_per_epoch=steps_per_epoch_train,
                                             validation_steps=steps_per_epoch_val,
                                             callbacks=[reduce_lr, early_stop])
    #Add epoch numbers
    history_mlp.history['epoch'] = list(range(1, len(history_mlp.history['loss'])+1))

    with open(f'{args.outputFolder}{ae_string}_MLP_Training_history_{MLP_neurons_str}_Batch_{args.batchsize}_SEED_{args.seed}.pkl', 'wb') as file_pi:
        pickle.dump(history_mlp.history, file_pi)

    print(get_time() + "Training finished for MLP...", )

    plot_loss_graph(train_loss=history_mlp.history['loss'],
                    val_loss=history_mlp.history['val_loss'],

                    f1_mean=history_mlp.history['f1_mean'],
                    val_f1_mean=history_mlp.history['val_f1_mean'],
                    precision_mean=history_mlp.history['precision_mean'],
                    val_precisison_mean=history_mlp.history['val_precision_mean'],
                    recall_mean=history_mlp.history['recall_mean'],
                    val_recall_mean=history_mlp.history['val_recall_mean'],

                    f1_disease=history_mlp.history['f1_disease'],
                    val_f1_disease=history_mlp.history['val_f1_disease'],
                    precision_disease=history_mlp.history['precision_disease'],
                    val_precision_disease=history_mlp.history['val_precision_disease'],
                    recall_disease=history_mlp.history['recall_disease'],
                    val_recall_disease=history_mlp.history['val_recall_disease'],

                    f1_species=history_mlp.history['f1_species'],
                    val_f1_species=history_mlp.history['val_f1_species'],
                    precision_species=history_mlp.history['precision_species'],
                    val_precision_species=history_mlp.history['val_precision_species'],
                    recall_species=history_mlp.history['recall_species'],
                    val_recall_species=history_mlp.history['val_recall_species'],

                    f1_celltype=history_mlp.history['f1_celltype'],
                    val_f1_celltype=history_mlp.history['val_f1_celltype'],
                    precision_celltype=history_mlp.history['precision_celltype'],
                    val_precision_celltype=history_mlp.history['val_precision_celltype'],
                    recall_celltype=history_mlp.history['recall_celltype'],
                    val_recall_celltype=history_mlp.history['val_recall_celltype'],
                    save_path=args.outputFolder)

    loss_curve(history_mlp.history['epoch'],
               history_mlp.history['loss'],
               history_mlp.history['val_loss'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_Train_val_loss",
               ylim=[0.00, 0.551])

    loss_curve(history_mlp.history['epoch'],
               history_mlp.history['f1_mean'],
               history_mlp.history['val_f1_mean'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_F1_train_val_mean",
               ylim=[0.8,1.005])

    loss_curve(history_mlp.history['epoch'],
               history_mlp.history['f1_species'],
               history_mlp.history['val_f1_species'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_F1_train_val_species",
               ylim=[0.95,1.005])

    loss_curve(history_mlp.history['epoch'],
               history_mlp.history['f1_celltype'],
               history_mlp.history['val_f1_celltype'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_F1_train_val_celltype",
               ylim=[0.85,1.005])

    loss_curve(history_mlp.history['epoch'],
               history_mlp.history['f1_disease'],
               history_mlp.history['val_f1_disease'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_F1_train_val_disease",
               ylim=[0.85,1.005])

    loss_curve_prec_rec(history_mlp.history['epoch'],
               history_mlp.history['precision_mean'],
               history_mlp.history['val_precision_mean'],
               history_mlp.history['recall_mean'],
               history_mlp.history['val_recall_mean'],
               metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_prec_rec_train_val_mean",
               ylim=[0.8,1.005])

    loss_curve_prec_rec(history_mlp.history['epoch'],
               history_mlp.history['precision_species'],
               history_mlp.history['val_precision_species'],
               history_mlp.history['recall_species'],
               history_mlp.history['val_recall_species'],
               metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_prec_rec_train_val_species",
               ylim=[0.95,1.005])

    loss_curve_prec_rec(history_mlp.history['epoch'],
               history_mlp.history['precision_celltype'],
               history_mlp.history['val_precision_celltype'],
               history_mlp.history['recall_celltype'],
               history_mlp.history['val_recall_celltype'],
               metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_prec_rec_train_val_celltype",
               ylim=[0.85,1.005])

    loss_curve_prec_rec(history_mlp.history['epoch'],
               history_mlp.history['precision_disease'],
               history_mlp.history['val_precision_disease'],
               history_mlp.history['recall_disease'],
               history_mlp.history['val_recall_disease'],
               metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
               save_dir=f'{args.outputFolder}/Metric_figures',
               name="MLP_prec_rec_train_val_disease",
               ylim=[0.85,1.005])

    mlp_model_instance.save(
        args.outputFolder + ae_string + '_MLP_' + MLP_neurons_str + '_Batch_' + str(args.batchsize) + '_SEED_' + str(
            args.seed) + '.keras')
    print(get_time() + "MLP Saved!")

    ''' Load MLP
    mlp_model_instance = tf.keras.models.load_model(
        args.outputFolder + ae_string + '_MLP_' + MLP_neurons_str + '_Batch_' + str(args.batchsize) + '_SEED_' + str(
            args.seed) + '.keras',
        compile=True)
    '''
    # Testing
    if build_autoencoder is True:
        print(get_time() + "Testing Phase: Predict with autoencoder...", )
        pred_ae_test = autoencoder_instance.predict(dataset_test, steps=steps_per_epoch_test)
        print(get_time() + "Testing Phase: Prediction with autoencoder finished!", )
        print(get_time() + "Testing Phase: Predict with MLP...", )
        pred_Test = mlp_model_instance.predict(pred_ae_test)
        print(get_time() + "Testing Phase: Prediction with MLP finished!", )
    else:
        print(get_time() + "Testing Phase: Predict with MLP...", )
        pred_Test = mlp_model_instance.predict(dataset_test, steps=steps_per_epoch_test)
        print(get_time() + "Testing Phase: Prediction with MLP finished!", )

    np.savetxt(
        args.outputFolder + ae_string + '_pred_raw_results' + '_MLP_' + MLP_neurons_str + '_Batch_' + str(
            args.batchsize) + '_SEED_' + str(
            args.seed) + '.csv', pred_Test, delimiter=',')

    pred_result = onehot_predict(pred_Test)
    np.savetxt(
        args.outputFolder + ae_string + '_pred_results' + '_MLP_' + MLP_neurons_str + '_Batch_' + str(
            args.batchsize) + '_SEED_' + str(
            args.seed) + '.csv', pred_result, delimiter=',')


if __name__ == "__main__":
    main()






