import argparse, pickle
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import MultiLabelBinarizer
from UP_preprocess import MultiLabelBin, createDatasets
from UP_models import DEAutoencoder, MLP_Model
from UP_utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Args:
    pass
args = Args()
args.trainpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/RANDOMIZED_250819_koenig_train_set_p_80.csv'
args.valpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/RANDOMIZED_250819_koenig_val_set_p_20.csv'
args.trainlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/RANDOMIZED_250819_koenig_train_set_label_p_80.csv'
args.vallabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/RANDOMIZED_250819_koenig_val_set_label_p_20.csv'
args.testpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/250819_koenig_test_set.csv'
args.testlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/250819_koenig_test_set_label.csv'
args.outputFolder = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250813_publicdata_analysis/koenig_et_al_2022/retrain/RUN2'
args.batchsize = 10240
args.seed = 1234
tf.random.set_seed(1234)
args.MLPLayers = 3
args.MLPNeurons = [795, 230, 105]
args.AutoLayers = 3
args.AutoNeurons = [5000, 2400, 350, 2200, 5150]


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


def createDatasets(trainpath, valpath, testpath, trainlabels, vallabels, testlabels, batchsize, build_autoencoder):
    """Creates Tensors out of the generators and returns them

    Args:
        trainpath(string): Location of training_data.csv
        valpath(string): Location of validation_data.csv
        testpath(string): Location of test_data.csv
        batch_size(int): Size of loaded batches

    Returns:
        dataset_train(FlatMapDataset): Tensor output (generator) for training data
        dataset_val(FlatMapDataset): Tensor output (generator) for validation data
        dataset_test(FlatMapDataset): Tensor output (generator) for test data

    """
    if build_autoencoder is True:
        dataset_train = tf.data.Dataset.from_generator(autoencoder_generator_wrapper,
                                                   args=[trainpath, batchsize],
                                                   output_signature=(
                                                       tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 28708), dtype=tf.float32)))

        dataset_val = tf.data.Dataset.from_generator(autoencoder_generator_wrapper,
                                                 args=[valpath, batchsize],
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 28708), dtype=tf.float32)))

        dataset_test = tf.data.Dataset.from_generator(autoencoder_generator_wrapper,
                                                  args=[testpath, batchsize],
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(None, 28708), dtype=tf.float32)))
    if build_autoencoder is False:
        dataset_train = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                       args=[trainpath, trainlabels, batchsize],
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

        dataset_val = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                     args=[valpath, vallabels, batchsize],
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

        dataset_test = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                      args=[testpath, testlabels, batchsize],
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, 28708), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

    return dataset_train, dataset_val, dataset_test

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

    """
    without generator, more memory usage but increased speed
    history_auto = autoencoder_instance.fit(dataset_train,dataset_train, epochs=500,
                                            validation_data=(dataset_val,dataset_val),
                                            validation_freq=1,
                                            batch_size=args.batchsize,
                                            callbacks=[reduce_lr, early_stop])
    """

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
mlp_model_instance.compile(optimizer='adam', loss="binary_focal_crossentropy", # was macro_f1_loss, the binary_focal_crossentropy is better fitting for the data set
                           metrics=[f1_mean, f1_species, f1_celltype, f1_disease,
                                    precision_mean, precision_species, precision_celltype, precision_disease,
                                    recall_mean, recall_species, recall_celltype, recall_disease,
                                    AUC(name='roc_auc', curve='ROC', multi_label=True, num_labels=13),
                                    AUC(name='pr_auc', curve='PR', multi_label=True, num_labels=13)], )

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

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['loss'],
           history_mlp.history['val_loss'],
           save_dir=f'{args.outputFolder}/Metric_figures',
           name="MLP_Train_val_loss",
           ylim=[0.00, 0.02])

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