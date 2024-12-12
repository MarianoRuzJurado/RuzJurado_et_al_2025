import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import tensorflow as tf
import pandas as pd
import numpy as np
import shap
import os
from keras import backend as K
from datetime import datetime
from scipy.stats import ranksums, ttest_ind, shapiro
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix


matplotlib.use('Agg')  # Use the 'Agg' backend which is non-interactive and doesn't require a display


def plot_loss_graph(train_loss, val_loss, save_path=None,
                    f1_mean=None, f1_species=None, f1_celltype=None, f1_disease=None,
                    recall_mean=None, recall_species=None, recall_celltype=None, recall_disease=None,
                    precision_mean=None, precision_species=None, precision_celltype=None, precision_disease=None,
                    val_f1_mean=None, val_f1_species=None, val_f1_celltype=None, val_f1_disease=None,
                    val_recall_mean=None, val_recall_species=None, val_recall_celltype=None, val_recall_disease=None,
                    val_precisison_mean=None, val_precision_species=None, val_precision_celltype=None, val_precision_disease=None,
                    ):
    """Plotting function for metrics used in training, creates plots for the given metrics in save directory


    Args:
        train_loss(key-value pair): Dictionary entry for the metric train_loss
        val_loss(key-value pair): Dictionary entry for the metric val_loss
        f1_mean(key-value pair): Dictionary entry for the metric f1_mean
        f1_species(key-value pair): Dictionary entry for the metric f1_species
        f1_celltype(key-value pair): Dictionary entry for the metric f1_celltype
        f1_disease(key-value pair): Dictionary entry for the metric f1_disease
        val_f1_mean(key-value pair): Dictionary entry for the metric val_f1_mean
        val_f1_species(key-value pair): Dictionary entry for the metric val_f1_species
        val_f1_celltype(key-value pair): Dictionary entry for the metric val_f1_celltype
        val_f1_disease(key-value pair): Dictionary entry for the metric val_f1_disease
        save_path(string): Path to the output Folder
    """


    # Extract the number of epochs
    epochs = range(1, len(train_loss) + 1)

    # Plot train/validation loss graph
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.ylim([0, 1])
    plt.xlabel('Epochs')
    plt.ylabel('F1-Loss')
    plt.legend(loc='lower right', fontsize='5')
    plt.title('Training & Validation F1-Loss')


    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path + "/Training_Validation_F1_loss.pdf")
        plt.clf()

    if f1_mean is not None:
        plt.plot(epochs, f1_mean, label='Training F1-Mean')
        plt.plot(epochs, val_f1_mean, label='Validation F1-Mean')
        plt.plot(epochs, recall_mean, label='Training Recall-Mean')
        plt.plot(epochs, val_recall_mean, label='Validation Recall-Mean')
        plt.plot(epochs, precision_mean, label='Training Precisison-Mean')
        plt.plot(epochs, val_precisison_mean, label='Validation Precisison-Mean')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('F1-,Recall- & Precision-Mean')
        plt.legend(loc='lower right', fontsize='5')
        plt.title('Training & Validation Metrics-Mean across all classes')


        if save_path:
            plt.savefig(save_path + "/Training_Validation_F1_Mean_all_classes.pdf")
            plt.clf()

    if f1_species is not None:
        plt.plot(epochs, f1_species, label='Training F1-Species')
        plt.plot(epochs, val_f1_species, label='Validation F1-Species')
        plt.plot(epochs, recall_species, label='Training Recall-Species')
        plt.plot(epochs, val_recall_species, label='Validation Recall-Species')
        plt.plot(epochs, precision_species, label='Training Precisison-Species')
        plt.plot(epochs, val_precision_species, label='Validation Precisison-Species')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('F1-,Recall- & Precision-Species')
        plt.legend(loc='lower right', fontsize='5')
        plt.title('Training & Validation Metrics-Species')


        if save_path:
            plt.savefig(save_path + "/Training_Validation_F1_Mean_species.pdf")
            plt.clf()

    if f1_celltype is not None:
        plt.plot(epochs, f1_celltype, label='Training F1-Celltype')
        plt.plot(epochs, val_f1_celltype, label='Validation F1-Celltype')
        plt.plot(epochs, recall_celltype, label='Training Recall-Celltype')
        plt.plot(epochs, val_recall_celltype, label='Validation Recall-Celltype')
        plt.plot(epochs, precision_celltype, label='Training Precisison-Celltype')
        plt.plot(epochs, val_precision_celltype, label='Validation Precisison-Celltype')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('F1-,Recall- & Precision-Celltype')
        plt.legend(loc='lower right', fontsize='5')
        plt.title('Training & Validation Metrics-Celltype')


        if save_path:
            plt.savefig(save_path + "/Training_Validation_F1_Mean_celltype.pdf")
            plt.clf()

    if f1_disease is not None:
        plt.plot(epochs, f1_disease, label='Training F1-Disease')
        plt.plot(epochs, val_f1_disease, label='Validation F1-Disease')
        plt.plot(epochs, recall_disease, label='Training Recall-Disease')
        plt.plot(epochs, val_recall_disease, label='Validation Recall-Disease')
        plt.plot(epochs, precision_disease, label='Training Precisison-Disease')
        plt.plot(epochs, val_precision_disease, label='Validation Precisison-Disease')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        plt.ylabel('F1-,Recall- & Precision-Disease')
        plt.legend(loc='lower right', fontsize='5')
        plt.title('Training & Validation Metrics-Disease')


        if save_path:
            plt.savefig(save_path + "/Training_Validation_F1_Mean_Disease.pdf")
            plt.clf()

def loss_curve(x_data,
               train_err, val_err,
               name='example',
               save_dir=None,
               ylim=[0,2]):

    mec1 = '#2F4F4F'
    mfc1 = '#C0C0C0'
    mec2 = 'maroon'
    mfc2 = 'pink'

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot(x_data, train_err, '-', color=mec1, marker='o',
            mec=mec1, mfc=mfc1, ms=4, alpha=0.5, label='train')
    ax.plot(x_data, val_err, '--', color=mec2, marker='s',
            mec=mec2, mfc=mfc2, ms=4, alpha=0.5, label='validation')

    max_val_err = max(val_err)
    ax.axhline(max_val_err, color='b', linestyle='--', alpha=0.3)

    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Loss (Units)')
    ax.set_ylim(ylim[0], ylim[1])

    ax.legend(loc=4, framealpha=0.35, handlelength=1.5)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)
    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_curve.pdf'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()

def loss_curve_prec_rec(x_data,
               train_err, val_err, train_err2, val_err2,
               name='example',
               metrics_name=None,
               save_dir=None,
               ylim=[0,2]):

    mec1 = '#2F4F4F'
    mfc1 = '#C0C0C0'
    mec2 = 'maroon'
    mfc2 = 'pink'

    mec3 = '#4b006e'
    mfc3 = '#C0C0C0'
    mec4 = '#ff7f0e'
    mfc4 = 'pink'

    if metrics_name is None:
        metrics_name = ["train1", "validation1", "train2", "validation2"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(x_data, train_err, '-', color=mec1, marker='o',
            mec=mec1, mfc=mfc1, ms=4, alpha=0.5, label=metrics_name[0])
    ax.plot(x_data, val_err, '--', color=mec2, marker='s',
            mec=mec2, mfc=mfc2, ms=4, alpha=0.5, label=metrics_name[1])
    max_val_err = max(val_err)
    ax.axhline(max_val_err, color='b', linestyle='--', alpha=0.3)

    ax.plot(x_data, train_err2, '-', color=mec3, marker='^',
            mec=mec3, mfc=mfc3, ms=4, alpha=0.5, label=metrics_name[2])
    ax.plot(x_data, val_err2, '-', color=mec4, marker='h',
            mec=mec4, mfc=mfc4, ms=4, alpha=0.5, label=metrics_name[3])
    max_val_err2 = max(val_err2)
    ax.axhline(max_val_err2, color='b', linestyle='--', alpha=0.3)


    ax.set_xlabel('Number of training epochs')
    ax.set_ylabel('Loss (Units)')
    ax.set_ylim(ylim[0], ylim[1])

    ax.legend(loc=4, framealpha=0.35, handlelength=1.5)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True,
                   top=True,
                   direction='in',
                   length=7)
    ax.tick_params(which='minor',
                   right=True,
                   top=True,
                   direction='in',
                   length=4)
    if save_dir is not None:
        fig_name = f'{save_dir}/{name}_curve.pdf'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()

def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S: ')
    return formatted_time


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

def onehot_predict(y_pred):
    """Get the top hits for our classification after predicting

    Args:
        y_pred(array): Predicted label by MLP

    Returns:
        y_bin(array): Top three hits in the matrix per row from y_pred
    """
    # Get indices of the top 3 values in each row
    top_indices = np.argsort(-y_pred)[:, :3]

    # Initialize an output matrix of zeros with the same shape as y_pred
    y_bin = np.zeros(y_pred.shape)

    # Set the positions of the top 3 values in each row to 1
    for row in range(y_pred.shape[0]):
        y_bin[row, top_indices[row]] = 1

    return y_bin

def f1_score(y_true, y_pred):
    """Definition of the micro f1_score, calculation across all classes

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        F1-Score(tensor): micro F1-Score calculated over all class
    """
    # Define the true positives, false positives and false negatives
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    # Calculate the precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Calculate the F1 score
    f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    return f1_score


def f1_score_collector(y_true, y_pred):
    """Definition of the macro f1_score, calculation for each classes

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        f1_score_collector(list): macro F1-Score calculated for each class stored in a list
    """
    # Define the true positives, false positives and false negatives
    f1_score_collector = []
    for column in range(0, y_true.shape[1]):
        tp = K.sum(K.round(K.clip(y_true[:, column] * y_pred[:, column], 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred[:, column] - y_true[:, column], 0, 1)))
        fn = K.sum(K.round(K.clip(y_true[:, column] - y_pred[:, column], 0, 1)))

        # Calculate the precision and recall
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        # Calculate the F1 score
        f1_score_collector.append(2 * ((precision * recall) / (precision + recall + K.epsilon())))

    return f1_score_collector

@tf.keras.saving.register_keras_serializable()
def f1_mean(y_true, y_pred):
    """Calculates the mean macro f1_score, uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector)(tensor): mean macro F1-Score over all classes
    """
    collector = f1_score_collector(y_true, y_pred)
    return tf.reduce_mean(collector)

@tf.keras.saving.register_keras_serializable()
def f1_species(y_true, y_pred):
    """Calculates the mean macro f1_score for species (first two classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[:1])(tensor): mean macro F1-Score over species classes
    """
    collector = f1_score_collector(y_true, y_pred)
    return tf.reduce_mean(collector[:1])

@tf.keras.saving.register_keras_serializable()
def f1_celltype(y_true, y_pred):
    """Calculates the mean macro f1_score for celltype (2:8 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce.mean(collector[2:8])(tensor): mean macro F1-Score over celltype classes
    """
    collector = f1_score_collector(y_true, y_pred)
    return tf.reduce_mean(collector[2:8])

@tf.keras.saving.register_keras_serializable()
def f1_disease(y_true, y_pred):
    """Calculates the mean macro f1_score for disease (9:12 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[9:12])(tensor): mean macro F1-Score over disease classes
    """
    collector = f1_score_collector(y_true, y_pred)
    return tf.reduce_mean(collector[9:12])

@tf.keras.saving.register_keras_serializable()
def macro_f1_loss(y_true, y_pred):
    """Calculates the average macro f1-loss for each class and applies a mean on them

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        macro_cost(tensor): mean macro F1-loss (1-F1_macro_average)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum(y_pred, axis=0) - tp
    fn = tf.reduce_sum(y_true, axis=0) - tp

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1_scores = 2 * (precision * recall) / (precision + recall + K.epsilon())
    cost = 1 - f1_scores
    macro_cost = tf.reduce_mean(cost)

    return macro_cost

def precision_collector(y_true, y_pred):
    """Definition of the macro precision_score, calculation for each classes

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        precision_collector(list): macro precision-Score calculated for each class stored in a list
    """
    # Define the true positives, false positives and false negatives
    precision_collector = []
    for column in range(0, y_true.shape[1]):
        tp = K.sum(K.round(K.clip(y_true[:, column] * y_pred[:, column], 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred[:, column] - y_true[:, column], 0, 1)))
        fn = K.sum(K.round(K.clip(y_true[:, column] - y_pred[:, column], 0, 1)))

        # Calculate the precision and recall
        precision_collector.append(tp / (tp + fp + K.epsilon()))


    return precision_collector

@tf.keras.saving.register_keras_serializable()
def precision_mean(y_true, y_pred):
    """Calculates the mean macro f1_score, uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector)(tensor): mean macro F1-Score over all classes
    """
    collector = precision_collector(y_true, y_pred)
    return tf.reduce_mean(collector)

@tf.keras.saving.register_keras_serializable()
def precision_species(y_true, y_pred):
    """Calculates the mean macro f1_score for species (first two classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[:1])(tensor): mean macro F1-Score over species classes
    """
    collector = precision_collector(y_true, y_pred)
    return tf.reduce_mean(collector[:1])

@tf.keras.saving.register_keras_serializable()
def precision_celltype(y_true, y_pred):
    """Calculates the mean macro f1_score for celltype (2:8 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce.mean(collector[2:8])(tensor): mean macro F1-Score over celltype classes
    """
    collector = precision_collector(y_true, y_pred)
    return tf.reduce_mean(collector[2:8])

@tf.keras.saving.register_keras_serializable()
def precision_disease(y_true, y_pred):
    """Calculates the mean macro f1_score for disease (9:12 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[9:12])(tensor): mean macro F1-Score over disease classes
    """
    collector = precision_collector(y_true, y_pred)
    return tf.reduce_mean(collector[9:12])


def recall_collector(y_true, y_pred):
    """Definition of the macro recall_score, calculation for each classes

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        recall_collector(list): macro recall-Score calculated for each class stored in a list
    """
    # Define the true positives, false positives and false negatives
    recall_collector = []
    for column in range(0, y_true.shape[1]):
        tp = K.sum(K.round(K.clip(y_true[:, column] * y_pred[:, column], 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred[:, column] - y_true[:, column], 0, 1)))
        fn = K.sum(K.round(K.clip(y_true[:, column] - y_pred[:, column], 0, 1)))

        # Calculate the precision and recall
        recall_collector.append(tp / (tp + fn + K.epsilon()))


    return recall_collector

@tf.keras.saving.register_keras_serializable()
def recall_mean(y_true, y_pred):
    """Calculates the mean macro f1_score, uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector)(tensor): mean macro F1-Score over all classes
    """
    collector = recall_collector(y_true, y_pred)
    return tf.reduce_mean(collector)

@tf.keras.saving.register_keras_serializable()
def recall_species(y_true, y_pred):
    """Calculates the mean macro f1_score for species (first two classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[:1])(tensor): mean macro F1-Score over species classes
    """
    collector = recall_collector(y_true, y_pred)
    return tf.reduce_mean(collector[:1])

@tf.keras.saving.register_keras_serializable()
def recall_celltype(y_true, y_pred):
    """Calculates the mean macro f1_score for celltype (2:8 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce.mean(collector[2:8])(tensor): mean macro F1-Score over celltype classes
    """
    collector = recall_collector(y_true, y_pred)
    return tf.reduce_mean(collector[2:8])

@tf.keras.saving.register_keras_serializable()
def recall_disease(y_true, y_pred):
    """Calculates the mean macro f1_score for disease (9:12 classes), uses the f1_score_collector function

    Args:
        y_true(array): Ground-truth labels of data
        y_pred(array): Predicted labels of data

    Returns:
        tf.reduce_mean(collector[9:12])(tensor): mean macro F1-Score over disease classes
    """
    collector = recall_collector(y_true, y_pred)
    return tf.reduce_mean(collector[9:12])


def beeswarm_gene_per_class(dataset_test, gene, shap_values, output): # save beeswarm with this function putting plots below each other, kinda ugly.... USE BASH SCRIPT pdfjam_combiner.sh in beeswarm folder
    """
    :param dataset_test: Test data set shap values where calculated with
    :param gene: gene to make beeswarm plot per class
    :param shap_values: retrieved shap values from SHAP explainer
    :param output: define plot name and output folder
    :return:
    """
    # Get relevant columns of dataset_test for the provided gene
    dataset_test_sub = dataset_test[[gene]]
    dataset_test_genes_sub = [gene]
    fig = plt.figure()

    # create beeswarm
    for i in range(len(shap_values)):
        shap_values_loc = [dataset_test.columns.get_loc(gene) for gene in dataset_test_genes_sub]
        shap_values_sub = shap_values[i][:, shap_values_loc]

        axes = plt.subplot2grid(shape=(13, 1), loc=(i, 0), fig=fig)
        shap.summary_plot(shap_values_sub, features=dataset_test_sub, feature_names=dataset_test_genes_sub,
                          max_display=50, show=False, plot_size=(5, 10))

        ax = plt.gca()
        ax.set_xlabel('')

    plt.savefig(output, bbox_inches='tight')
    plt.close()

def simpleclass(adata, shap_class, pattern_1, pattern_2, ground_truth, ground_truth_exp, outDir, testing_method="t", SHAP_thresh=0.01, logfc_thresh=0.1, expression_threshold_test_data=0.1, p_adj_thresh=0.05, sparse_matrix=False):
    """
    :param adata: AnnData object with SHAP column 'Class' in obs
    :param shap_class: SHAP class of relevance for testing
    :param pattern_1: pattern for cells in group 1 defining Species, Celltype, Disease e.g. [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] Human CM HFrEF
    :param pattern_2: pattern for cells in group 2 defining Species, Celltype, Disease
    :param ground_truth: Path to dataframe with ground truth labels for cells
    :param ground_truth_exp: Expression of ground truth data, used for defining inverse or direct predictor
    :param outDir: Specify Directory to save results
    :param testing_method: test to conduct, default "students" t-test, if non-parametric test preferred, use "wilcoxon"
    :param SHAP_thresh: Threshold SHAP-Value must reach in one of the two groups with its absolute value, default 0.0001
    :param logfc_thresh: Threshold logfoldchange between groups, verything below will be discarded
    :param expression_threshold_test_data: Threshold for minimum average expression ins test data
    :param p_adj_thresh: Threshold for p-adjusted values, everything below that will be saved
    :param sparse_matrix: BOOLEAN if data.X in sparse or dense format
    :return:
    """
    # Create folder to save results
    Path(f'{outDir}').mkdir(exist_ok=True)

    # Get positions of 1s in patterns, just relevant for clear names
    pos_pattern_1 = [index for index, value in enumerate(pattern_1) if value ==1]
    pos_pattern_2 = [index for index, value in enumerate(pattern_2) if value == 1]

    # Build Dataframe from .X and assign Gene names
    class_adata = adata[adata.obs['Class'] == shap_class]
    if sparse_matrix:
        class_adata_df = pd.DataFrame(class_adata.X.toarray())
    else:
        class_adata_df = pd.DataFrame(class_adata.X)
    class_adata_df.columns = class_adata.var_names

    # Build Ground Truth dataframes with clear names
    ground_truth_df = pd.read_csv(ground_truth, header=None, low_memory=False)
    test_labels = ('Human', 'Mice',
                   'Cardiomyocytes', 'Endothelial', 'Fibroblasts', 'Immune.cells', 'Neuro', 'Pericytes',
                   'Smooth.Muscle',
                   'AS', 'HFpEF', 'HFrEF', 'CTRL') # Assumption of same order in ground truth as always
    ground_truth_df.columns = test_labels

    # Make clear names
    clear_name_pattern1 = '_'.join(map(str, [test_labels[pos_pattern_1[0]], test_labels[pos_pattern_1[1]], test_labels[pos_pattern_1[2]]]))
    clear_name_pattern2 = '_'.join(map(str, [test_labels[pos_pattern_2[0]], test_labels[pos_pattern_2[1]], test_labels[pos_pattern_2[2]]]))

    # Group1
    mask_1 = (ground_truth_df == pattern_1).all(axis=1)
    mask_1.value_counts()
    class_adata_mask_1 = class_adata_df[mask_1]


    # Group2
    mask_2 = (ground_truth_df == pattern_2).all(axis=1)
    mask_2.value_counts()
    class_adata_mask_2 = class_adata_df[mask_2]

    # Get expression data for groups
    ground_truth_exp.reset_index(drop=True, inplace=True)
    ground_truth_exp_mask_1 = ground_truth_exp[mask_1]
    ground_truth_exp_mask_2 = ground_truth_exp[mask_2]

    # Generate ListTest (list of column indices for comparison)
    ListTest = [(i, i) for i in class_adata_mask_1.columns]

    if testing_method == 't':
        # Perform pairwise comparisons
        results = []
        for pair in ListTest:
            # print(pair)
            column_idx_a, column_idx_b = pair
            subset_a = class_adata_mask_1[column_idx_a]
            subset_b = class_adata_mask_2[column_idx_b]
            stat, p_val = ttest_ind(subset_a, subset_b, equal_var=True)
            results.append((column_idx_a, p_val, np.mean(subset_a), np.mean(subset_b)))

    if testing_method == 'wilcoxon':
        # Perform pairwise comparisons
        results = []
        for pair in ListTest:
            # print(pair)
            column_idx_a, column_idx_b = pair
            subset_a = class_adata_mask_1[column_idx_a]
            subset_b = class_adata_mask_2[column_idx_b]
            stat, p_val = ranksums(subset_a, subset_b)
            results.append((column_idx_a, p_val, np.mean(subset_a), np.mean(subset_b)))

    # Correct with benjamini-hochberg-Method
    results_p = [tup[1] for tup in results]
    rejected, p_adjusted, _, alpha_corrected = multipletests(results_p, alpha=0.05,
                                                             method='fdr_bh', is_sorted=False, returnsorted=False)
    # Add p-adjusted
    for i, result in enumerate(results):
        # print(i)
        # print(result)
        results[i] = result + (p_adjusted[i],)

    # epsilon = 1e-296  # Add a small epsilon value to prevent division by zero, right now no values are 0 with z-scaling, obsolete
    columns = ['Feature', 'p_val', 'mean_SHAP_group1', 'mean_SHAP_group2', 'p_adj']
    results_df = pd.DataFrame(results, columns=columns)
    results_df['avg_cube_root_FC'] = np.cbrt(results_df['mean_SHAP_group1'] - results_df['mean_SHAP_group2']) # Take the cube root since defined in neg pos and 0 values
    results_df = results_df[(results_df['p_adj'] <= p_adj_thresh)] # only significant
    results_df = results_df[(abs(results_df['mean_SHAP_group1']) >= SHAP_thresh) | (abs(results_df['mean_SHAP_group2']) >= SHAP_thresh)] # Filter by absolute Shap values
    results_df = results_df[(abs(results_df['avg_cube_root_FC']) >= logfc_thresh)] # subset by thresh
    results_df = results_df.sort_values(['avg_cube_root_FC'], ascending=False)

    print(get_time() + "Filtering DEGs by a minimum expression of " + str(expression_threshold_test_data) + " in test data set")
    results_df = expression_filter(DGE_result=results_df,
                                   expression_group1=ground_truth_exp_mask_1,
                                   expression_group2=ground_truth_exp_mask_2,
                                   expression_threshold_test_data=expression_threshold_test_data)

    # New Inverse and Direct marker calculation
    print("\n" + get_time() + "Starting predictor type analysis")
    results_df = predictor_type(DGE_result=results_df,
                             class_adata_mask_1=class_adata_mask_1,
                             expression_group=ground_truth_exp_mask_1)

    # Define direct or inverse predictor #DOESNT WORK CORRECTLY WE NEED THE EXPRESSION DATA
    '''results_df['predictor_type'] = np.where(
        ((results_df['mean_SHAP_group1'] < 0) & (results_df['mean_SHAP_group2'] < 0) & (results_df['mean_SHAP_group1'] > results_df['mean_SHAP_group2'])) |
        ((results_df['mean_SHAP_group1'] > 0) & (results_df['mean_SHAP_group2'] > 0) & (results_df['mean_SHAP_group1'] < results_df['mean_SHAP_group2']))
        ,'Inverse','Direct')'''

    results_df = results_df.dropna()
    results_df.to_excel(f'{outDir}/{testing_method}_test_on_class_{shap_class}_{clear_name_pattern1}_{clear_name_pattern2}.xlsx',
        index=False)

def predictor_type(DGE_result, class_adata_mask_1, expression_group):
    """
    :param DGE_result: DGE analysis result obtained by XAI-DGE analysis
    :param class_adata_mask_1: Shap values of cells of interest in SHAP group 1
    :param expression_group: expression data of test data set of cells in SHAP group 1
    :return: DGE_result with column containing marker type information
    """

    DGE_result['predictor_type'] = "NULL"
    for feature in tqdm(DGE_result['Feature'], desc='predictor_type_analysis'):

        shap_feature = class_adata_mask_1[feature]
        expr_feature = expression_group[feature]

        # Separate shap_features by higher and lower than 0
        shap_feature_pos = shap_feature[shap_feature > 0]
        shap_feature_neg = shap_feature[shap_feature < 0]

        # Get the expression of cells with pos and neg shap values
        expr_feature_pos = expr_feature.loc[shap_feature_pos.index]
        expr_feature_neg = expr_feature.loc[shap_feature_neg.index]

        # Add direct or inverse marker information based on which mean is higher
        predictor_type = "Direct" if np.mean(expr_feature_pos) > np.mean(expr_feature_neg) else "Inverse"

        # add the row in DGE_result
        DGE_result.loc[DGE_result['Feature'] == feature, 'predictor_type'] = predictor_type

    return DGE_result

def expression_filter(DGE_result, expression_group1, expression_group2, expression_threshold_test_data=0.1):
    """
    :param DGE_result: DGE analysis result obtained by XAI-DGE analysis
    :param expression_group: expression data of test data set of cells in SHAP group 1
    :param expression_group2: expression data of test data set of cells in SHAP group 1
    :param expression_threshold_test_data Threshold of minimum mean expression a gene must have in one of the groups in the expression data
    :return: DGE_result with column containing marker type information
    """

    # Hold to the features which do not pass the thresholding
    features_to_remove = []
    DGE_result['avg_expr_group1_testdata'] = "NULL"
    DGE_result['avg_expr_group2_testdata'] = "NULL"
    for feature in tqdm(DGE_result['Feature'], desc='filter_DEGs'):
        mean_expr_feature_1 = np.mean(expression_group1[feature])
        mean_expr_feature_2 = np.mean(expression_group2[feature])

        # Insert mean expr of group 1 and 2
        DGE_result.loc[DGE_result['Feature'] == feature, 'avg_expr_group1_testdata'] = mean_expr_feature_1
        DGE_result.loc[DGE_result['Feature'] == feature, 'avg_expr_group2_testdata'] = mean_expr_feature_2

        # Check if both are under threshold
        if mean_expr_feature_1 < expression_threshold_test_data or mean_expr_feature_2 < expression_threshold_test_data:
            features_to_remove.append(feature)

    DGE_result_filtered = DGE_result[~DGE_result['Feature'].isin(features_to_remove)]

    return DGE_result_filtered


# Write function beeswarm subset by cell type and disease
def beeswarm_on_subset(adata, species, cell_type, disease, ground_truth, data_set_test, outDir, genes=None, sparse_matrix=False):
    """
    :param adata: Anndata object with SHAP values
    :param species: Species to subset SHAP values on provide as tuple e.g. ("Human",)
    :param cell_type: Cell type to subset SHAP values on, works with more than one cell type, please provide as tuple
    :param disease: Disease to subset SHAP values on e.g. ("HFrEF",)
    :param ground_truth: Ground Truth Dataframe per cell for test data
    :param data_set_test: test data Dataframe with expression values
    :param outDir: Set output Folder
    :param genes: Group of genes to be depicted in beeswarm, if None Top 50 features
    :param sparse_matrix: BOOLEAN if data.X in sparse or dense format
    :return:
    """

    # Subset adata by cells of interest
    adata_sub = adata[adata.obs['Disease'].isin(disease)].copy()
    adata_sub = adata_sub[adata_sub.obs['Cell_type'].isin(cell_type)].copy()
    adata_sub = adata_sub[adata_sub.obs['Species'].isin(species)].copy()

    # Saving strings
    string_species = '_'.join(map(str, species))
    string_cell_type = '_'.join(map(str, cell_type))
    string_disease = '_'.join(map(str, disease))

    if isinstance(data_set_test, str):
        # Read in dataset_test
        print('Path to dataset provided: Reading in test dataset...')
        data_set_test = pd.read_csv(data_set_test, skiprows=0, header=0, index_col=0, low_memory=False)
        print('Reading in test dataset finished!')

    # Reconstruct SHAP DF from anndata object, split by class into numpy arrays
    unique_class = adata_sub.obs['Class'].unique()

    shap_values = []
    # Split adata by class write .X into list
    for class_val in unique_class:
        adata_sub_split = adata_sub[adata_sub.obs['Class'] == class_val].copy()
        if sparse_matrix:
            shap_values.append(adata_sub_split.X.toarray())
        else:
            shap_values.append(adata_sub_split.X)

    # Subset SHAP arrays by provided genes if not take top 50
    if genes is None:
        dataset_test_genes = data_set_test.columns.tolist()
        string_genes = 'Top50_sorted'
    else:
        dataset_test_genes = genes
        string_genes = f'{"_".join(map(str, genes))}_sorted'
        shap_values_loc = [data_set_test.columns.get_loc(gene) for gene in dataset_test_genes]
        shap_values = [df[:, shap_values_loc] for df in shap_values]

    # Build Ground Truth dataframes with clear names
    ground_truth_df = pd.read_csv(ground_truth, header=None, low_memory=False)
    test_labels = ('Human', 'Mice',
                   'Cardiomyocytes', 'Endothelial', 'Fibroblasts', 'Immune.cells', 'Neuro', 'Pericytes',
                   'Smooth.Muscle',
                   'AS', 'HFpEF', 'HFrEF', 'CTRL')  # Assumption of same order in ground truth as always
    ground_truth_df.columns = test_labels

    # Create 0 list which will be filled with 1s to represent pattern of relevant cells
    pattern = [0] * len(test_labels)
    # Replace zeros based on the arguments
    for i, label in enumerate(test_labels):
        print(i, label)
        if label in species or label in cell_type or label in disease:
            pattern[i] = 1

    splits_pat = split_pattern(pattern)

    # Convert sublists to tuples if more than one pattern, else use just this one
    if all(isinstance(sublist, list) for sublist in splits_pat):
        splits_pat_tuples = [tuple(sublist) for sublist in splits_pat]
        # Create a mask
        mask = ground_truth_df.apply(lambda row: tuple(row) in splits_pat_tuples, axis=1)
    else:
        mask = (ground_truth_df == splits_pat).all(axis=1)

    print(f'Number of cells labeled {string_species}_{string_cell_type}_{string_disease} in Test data : {mask.astype(int).sum()}') # Convert to boolean
    data_set_test.reset_index(drop=True, inplace=True)
    dataset_test_sub = data_set_test[mask]

    # Subset dataset_test_sub by the provided genes if provided
    if genes is not None:
        dataset_test_sub = dataset_test_sub[genes]

    # Create folder to save beeswarms
    Path(f'{outDir}_{string_genes}').mkdir(exist_ok=True)


    for i in range(len(shap_values)):
        shap.summary_plot(shap_values[i], features=dataset_test_sub, feature_names=dataset_test_genes, max_display=50)
        plt.savefig(f'{outDir}_{string_genes}/beeswarm_class_{i}_{string_species}_{string_cell_type}_{string_disease}_{string_genes}.pdf')
        plt.close()

def split_pattern(pattern):
    # Find the indices where '1' occurs in the pattern
    one_indices = [i for i, val in enumerate(pattern) if val == 1]

    # If there are 3 '1's, return the original pattern
    if len(one_indices) == 3:
        return pattern

    species_part = pattern[0:2]
    cell_part = pattern[2:9]
    disease_part = pattern[9:13]

    # Initialize an empty list to store the sublists

    sublist_cell = []
    # Iterate over the cell list
    for i, value in enumerate(cell_part):
        # print(i, value)
        # Create a sublist with zeros
        sublist = [0] * len(cell_part)
        # Place '1' at the index where the original list has '1'
        sublist[i] = value
        sublist = species_part + sublist
        sublist = sublist + disease_part
        sublist_cell.append(sublist)
    return sublist_cell


def z_score_per_cell(adata): # use the loop, here it might crash
    """
    :param adata: Anndata object with values to normalize (not a View of Anndata, use .copy() for sub  Anndata's)
    :return: Anndata object with z-score normalized values
    """

    adata_array = adata.X.toarray()

    # Calculate the mean and standard deviation of each gene
    mean_per_cell = np.mean(adata_array, axis=1)
    std_per_cell = np.std(adata_array, axis=1)

    # Add a small epsilon value to prevent division by zero
    epsilon = 1e-108
    std_per_cell = np.where(std_per_cell == 0, epsilon, std_per_cell)

    # Z-normalize the data for each feature
    z_normalized_data = (adata_array - mean_per_cell[:, np.newaxis]) / std_per_cell[:, np.newaxis] # includes broadcasting into 2D numpys for mean and std
    adata.X = z_normalized_data # no sparse since nothing is 0 anyways after z-scoring -> dense will actually be smaller
    return adata

def z_score_loop_per_cell(adata, sparse=False):
    """
    :param adata: Anndata object with values to normalize (not a View of Anndata, use .copy() for sub  Anndata's)
    :param sparse: for anndata.X is in sparse format or in dense
    :return: Anndata object with z-score normalized values
    """

    # Get row length of adata for loop
    row_numbers  = adata.X.shape[0]
    col_numbers = adata.X.shape[1]

    array = np.empty((row_numbers, col_numbers), dtype=float)
    print(get_time() + "Start calculating Z_scores...")
    if sparse is True:
        for i in tqdm(range(0, row_numbers), desc='z_score_loop_per_cell'):
            adata_array = adata.X[i].toarray()

            # Calculate the mean and standard deviation of each gene
            mean_per_cell = np.mean(adata_array, axis=1)
            std_per_cell = np.std(adata_array, axis=1)

            # Add a small epsilon value to prevent division by zero
            epsilon = 1e-108
            std_per_cell = np.where(std_per_cell == 0, epsilon, std_per_cell)

            # Z-normalize the data for each feature
            z_normalized_data = (adata_array - mean_per_cell[:, np.newaxis]) / std_per_cell[:, np.newaxis] # includes broadcasting into 2D numpys for mean and std
            array[i] = z_normalized_data

    if sparse is False:
        for i in tqdm(range(0, row_numbers), desc='z_score_loop_per_cell'):
            adata_array = adata.X[i]

            # Calculate the mean and standard deviation of each gene
            mean_per_cell = np.mean(adata_array)
            std_per_cell = np.std(adata_array)

            # Add a small epsilon value to prevent division by zero
            epsilon = 1e-108
            if std_per_cell == 0:
                std_per_cell = epsilon

            # Z-normalize the data for each feature
            z_normalized_data = (adata_array - mean_per_cell) / std_per_cell
            array[i] = z_normalized_data

    adata.X = array # no sparse since nothing is 0 anyways after z-scoring -> dense will actually be smaller
    print(get_time() + "Finished calculating Z_scores!")
    return adata


def shapiro_per_cell(adata, sparse_matrix=False, normalized=True):
    """
    :param adata: Anndata object containing cells with values to test for normality cell wise
    :param sparse_matrix: BOOLEAN if data.X in sparse or dense format
    :param normalized: BOOLEAN if already z-scored or not, default True
    :return: test statistic and summary
    """

    if not normalized:
        adata = z_score_per_cell(adata)

    if sparse_matrix:
        adata = adata.X.toarray() # Retrieve numpy array out of anndata
    else:
        adata = adata.X

    # Create an empty list to store the results of the Shapiro-Wilk test
    normality_results = []

    # Iterate over each row in the array
    for row in adata:
        # print(row)
        # Perform Shapiro-Wilk test on the current row
        stat, p = shapiro(row)

        # Append the result of the test to the list
        normality_results.append((stat, p))

    # Initialize counters for values below and above 0.05
    below_005_count = 0
    above_005_count = 0

    # Number of hypothesis tests (number of rows in the data)
    num_tests = len(normality_results)

    # Bonferroni corrected p-Value
    p_corrected = 0.05 / num_tests

    # Iterate over the results
    for stat, p in normality_results:
        if p <= p_corrected:
            below_005_count += 1
        else:
            above_005_count += 1

    # Print the summary
    print("Number of p-values below or equal to 0.05:", below_005_count)
    print("Number of p-values above 0.05:", above_005_count)

def free_memory():
    import gc
    import ctypes
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    return

def next_pattern(pattern_1, pattern_2):
    new_pattern_1 = pattern_1[:]  # make new pattern for first
    new_pattern_2 = pattern_2[:]  # make new pattern for second

    # Find the positions of the second and third '1' in pattern_1
    second_one_index_1 = new_pattern_1.index(1, 2)
    third_one_index_1 = new_pattern_1.index(1, second_one_index_1 + 1)

    # Move the third '1' in pattern_1 to the right
    if third_one_index_1 < len(new_pattern_1) - 2:  # careful with python start at 0
        new_pattern_1[third_one_index_1] = 0
        new_pattern_1[third_one_index_1 + 1] = 1
    else:
        # If the third '1' is at the second last position, reset it and move the second '1'
        if second_one_index_1 < len(new_pattern_1) - 3:
            new_pattern_1[second_one_index_1] = 0
            new_pattern_1[second_one_index_1 + 1] = 1
            new_pattern_1[9] = 1  # Reset the third '1' to position 9
            new_pattern_1[third_one_index_1] = 0

    # Update the second '1' in pattern_2 to match the new position of the second '1' in pattern_1
    second_one_index_2 = new_pattern_2.index(1, 2)
    if third_one_index_1 >= len(new_pattern_1) - 2:  # check for the position after moving the second "1" in pattern 1
        new_pattern_2[second_one_index_2] = 0
        new_pattern_2[second_one_index_1 + 1] = 1

    return new_pattern_1, new_pattern_2





