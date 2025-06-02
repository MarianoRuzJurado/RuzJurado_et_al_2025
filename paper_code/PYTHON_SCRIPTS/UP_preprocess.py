import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from UP_utils import get_time


def MultiLabelBin(label_path):
    """Defines a Multi-one-hot-encoded representation for the labels

    Args:
        label_path(string): Location of label.csv

    Returns:
        labels_bin(array): Numpy array of Multi-one-hot-encoded labels
    """
    print(get_time() + "Read Labels...", )
    label_path_df = pd.read_csv(label_path)
    labels = label_path_df.values.flatten().tolist()
    formatted_labels = [s.replace('_', '-').split('-') for s in labels]

    # Fit the multi-label Binarizer on the training set
    label_order = ['Human', 'Mice',
                   'Cardiomyocytes', 'Endothelial', 'Fibroblasts', 'Immune.cells', 'Neuro', 'Pericytes',
                   'Smooth.Muscle',
                   'AS', 'HFpEF', 'HFrEF', 'CTRL']
    mlb = MultiLabelBinarizer(classes=label_order)
    mlb.fit(formatted_labels)

    # Loop over all labels and show them
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i, label))

    labels_bin = mlb.transform(formatted_labels)
    return labels_bin


def autoencoder_generator(data_path, batch_size=32):
    """Loads in Data in batches infinitely for the training of the autoenoder

    Args:
        data_path(string): Location of training_data.csv
        batch_size(int): Size of loaded batches

    Yields:
        batch_data(array): Numpy array of the current batch
    """
    while True:  # add this line to create an infinite loop
        iterator = pd.read_csv(data_path, skiprows=0, header=0, verbose=False, chunksize=batch_size, index_col=0,
                               iterator=True, low_memory=False)
        for chunk in iterator:
            batch_data = chunk.values
            yield batch_data, batch_data


def autoencoder_generator_wrapper(data_path_bytes, batch_size):
    """PATH needs to be decoded for tensorflow, small wrapper

    Args:
        data_path_bytes(string): Location of training_data.csv
        batch_size(int): Size of loaded batches

    Returns:
        autoencoder_generator: Generator with now readable data_path for tf
    """
    data_path = data_path_bytes.decode('utf-8')
    return autoencoder_generator(data_path, batch_size)


def model_generator(data_path, labels, batch_size=32):
    """Loads in Data and Labels in batches infinitely for the training of the MLP

    Args:
        data_path(string): Location of training_data.csv
        labels(array): Numpy Array with one hot encoded labels for training data
        batch_size(int): Size of loaded batches

    Yields:
        batch_data(array): Numpy array of the current batch
        batch_labels(array): Numpy array of the corresponding labels
    """
    while True:  # add this line to create an infinite loop
        iterator = pd.read_csv(data_path, skiprows=0, header=0, verbose=False, chunksize=batch_size, index_col=0,
                               iterator=True, low_memory=False)

        idx = 0  # Track the current index
        for chunk in iterator:
            batch_data = chunk.values
            batch_labels = labels[idx: idx + batch_size]
            idx += batch_size
            yield batch_data, batch_labels


def model_generator_wrapper(data_path_bytes, labels, batch_size):
    """PATH needs to be decoded for tensorflow, small wrapper

    Args:
        data_path_bytes(string): Location of training_data.csv
        labels(array): Numpy Array with one hot encoded labels for training data
        batch_size(int): Size of loaded batches

    Returns:
        model_generator: Generator with now readable data_path for tf
    """
    data_path = data_path_bytes.decode('utf-8')
    return model_generator(data_path, labels, batch_size)


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
                                                       tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(None, 16545), dtype=tf.float32)))

        dataset_val = tf.data.Dataset.from_generator(autoencoder_generator_wrapper,
                                                 args=[valpath, batchsize],
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, 16545), dtype=tf.float32)))

        dataset_test = tf.data.Dataset.from_generator(autoencoder_generator_wrapper,
                                                  args=[testpath, batchsize],
                                                  output_signature=(
                                                      tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                      tf.TensorSpec(shape=(None, 16545), dtype=tf.float32)))
    if build_autoencoder is False:
        dataset_train = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                       args=[trainpath, trainlabels, batchsize],
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

        dataset_val = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                     args=[valpath, vallabels, batchsize],
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

        dataset_test = tf.data.Dataset.from_generator(model_generator_wrapper,
                                                      args=[testpath, testlabels, batchsize],
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, 16545), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, 13), dtype=tf.float32)))

    return dataset_train, dataset_val, dataset_test
