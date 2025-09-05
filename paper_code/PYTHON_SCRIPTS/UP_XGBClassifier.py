import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import MultiLabelBinarizer
import os
from datetime import datetime

"""Example of dataset
X, y = make_multilabel_classification(
    n_samples=32, n_classes=5, n_labels=3, random_state=0
)
"""

def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S: ')
    return formatted_time

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

class Args:
    pass
args = Args()
args.trainpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/RANDOMIZED_train_set_p_80.csv'
args.valpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/RANDOMIZED_val_set_p_20.csv'
args.trainlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/RANDOMIZED_train_set_labels_p_80.csv'
args.vallabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/RANDOMIZED_val_set_labels_p_20.csv'
args.testpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/240519_test_set_only_n2_samples.csv'
args.testlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/Storage_backup/anndata_shap/whole_matrix_Seurat_extract/240519_test_set_only_n2_samples_labels.csv'

dataset_train = pd.read_csv(args.trainpath, skiprows=0, header=0, index_col=0,low_memory=False)
dataset_val = pd.read_csv(args.valpath, skiprows=0, header=0, index_col=0,low_memory=False)
dataset_train = dataset_train.values
dataset_val = dataset_val.values

# Do the multi-label-hot encoding
train_labels_bin = MultiLabelBin(args.trainlabels)
val_labels_bin = MultiLabelBin(args.vallabels)
test_labels_bin = MultiLabelBin(args.testlabels)

clf = xgb.XGBClassifier(tree_method="hist",
                        eval_metric="logloss")

clf.fit(X=dataset_train,
        y=train_labels_bin,
        eval_set=[(dataset_val, val_labels_bin)],
        n_estimators=100,
        early_stopping_rounds=10)

clf.save_model('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/250806_xgb_CVD_model.json')

evals_result = clf.evals_result()
with open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/250806_xgb_CVD_val_history', 'wb') as f:
    pickle.dump(evals_result, f)

# Load test data and predict
dataset_test = pd.read_csv(args.testpath, skiprows=0, header=0, index_col=0,low_memory=False)


pred_result = clf.predict(dataset_test)

np.savetxt('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/XGBoost_pred_100epochs_logloss.csv', pred_result, delimiter=',')



########
#####
# With AE

train_labels_bin = MultiLabelBin(args.trainlabels)
val_labels_bin = MultiLabelBin(args.vallabels)
test_labels_bin = MultiLabelBin(args.testlabels)

ae_features_train = np.load('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/AE_features_train_5000_2400_350_2200_5150_Batch_10240_SEED_1234_RUN3.npy')
ae_features_val = np.load('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/AE_features_val_5000_2400_350_2200_5150_Batch_10240_SEED_1234_RUN3.npy')


clf = xgb.XGBClassifier(tree_method="hist",
                        eval_metric="logloss")

clf.fit(X=ae_features_train,
        y=train_labels_bin,
        eval_set=[(ae_features_val, val_labels_bin)])

clf.save_model('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/250806_xgb_AE_CVD_model.json')

evals_result = clf.evals_result()
with open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/250806_xgb_AE_CVD_val_history', 'wb') as f:
    pickle.dump(evals_result, f)

# Load test data and predict
dataset_test = pd.read_csv(args.testpath, skiprows=0, header=0, index_col=0,low_memory=False)


pred_result = clf.predict(dataset_test)

np.savetxt('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250806_XGBoost_training/XGBoost_AE_pred_100epochs_logloss.csv', pred_result, delimiter=',')

