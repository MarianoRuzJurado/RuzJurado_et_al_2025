import numpy as np
import pandas as pd

# import polars as pl
import pickle
from UP_utils import *
from UP_models import MLP_Model
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import random
import scanpy as sc
import anndata as ad


matplotlib.use('Agg')
plt.ion()

# train data
trainpath = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_train_set_without_Mice_AS_Neuro_p_80.csv'
dataset_train = pd.read_csv(trainpath, skiprows=0, header=0, index_col=0, low_memory=False)
#dataset_train = dataset_train.values

# Test data
testpath = '/media/Storage/anndata_shap/whole_matrix_Seurat_extract/240830_SHAP_set_balanced_200_per_celltype.csv'
dataset_test = pd.read_csv(testpath, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_test_genes = dataset_test.columns.tolist()
dataset_test = dataset_test.values
# dataset_labels_path = '/media/Helios_scStorage/Mariano/NN_Human_Mice/whole_matrix_Seurat_extract/RANDOMIZED_train_set_without_n2_samples_and_Mice_AS_Neuro_labels_p_80.csv'
# train_labels_bin = MultiLabelBin(dataset_labels_path)

mlp_model_instance = tf.keras.models.load_model(
    '/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/AE_True_5000_2400_350_2200_5150_MLP_795_230_105_Batch_10240_SEED_1234_RUN3.keras',
    custom_objects={'MLP_Model': MLP_Model}, compile=True)

# shap.DeepExplainer
random.seed(1111) # set seed 1111, consistence
#
random_set = np.random.choice(dataset_train.shape[0], 1000, replace=False)
background = dataset_train.iloc[random_set]

# save & load background as pickle
with open(
        '/media/Storage/anndata_shap/backgrounds/background_DeepExplainer_1000_cells.pkl',
        'wb') as f:
    pickle.dump(background, f)
background = pickle.load(
    open(
        '/media/Storage/anndata_shap/backgrounds/background_DeepExplainer_1000_cells.pkl',
        'rb'))

background = background.values
print(get_time() + "DeepExplainer : Start shap value calculation...")
explainer = shap.DeepExplainer((mlp_model_instance.layers[0].input, mlp_model_instance.layers[-1].output), background)
shap_values = explainer.shap_values(dataset_test)
print(get_time() + "DeepExplainer : Shap value calculation finished!")

# save as pickle

with open('/media/Storage/anndata_shap/background_200_balanced_data/240830_SHAP_set_balanced_200_per_celltype_1000_background.pkl', 'wb') as f:
    pickle.dump(shap_values, f)
shap_values = pickle.load(
    open('/media/Storage/anndata_shap/background_200_balanced_data/240830_SHAP_set_balanced_200_per_celltype_1000_background.pkl', 'rb'))

testpath = '/media/Storage/anndata_shap/whole_matrix_Seurat_extract/240830_SHAP_set_balanced_200_per_celltype.csv'
dataset_test = pd.read_csv(testpath, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_test_genes = dataset_test.columns.tolist()
# data_set_test = pd.read_csv('/media/Storage/R/SHAP/test_set_only_n2_samples.csv', skiprows=0, header=0, index_col=0, low_memory=False)


# save beeswarm per class
for i in range(len(shap_values)):
    shap.summary_plot(shap_values[i],
                      features=dataset_test,
                      feature_names=dataset_test_genes,
                      max_display=10,
                      cmap=shap.plots._utils.convert_color('matplotlib'),# convert_color function is modified
                      plot_size=(5.5,5)) # size for 10 genes: 5.5, 5
    plt.savefig('/media/Storage/anndata_shap/background_200_balanced_data/beeswarm_class_0' + str(
        i) + '_class_top10_features.pdf')
    plt.close()



# Define a directory to save the files
save_dir = "/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/shap_analysis/anndata_objects/background_100"
# save_dir = "/media/Storage/R/SHAP/objects/Z_background_100"

for i, class_array in enumerate(shap_values):
    print(i)
    df_class_array = pd.DataFrame(class_array)
    df_class_array.columns = dataset_test.columns
    df_class_array.index = dataset_test.index
    adata = sc.AnnData(df_class_array)
    adata.X = csr_matrix(adata.X)
    # Construct the full file path
    file_path = os.path.join(save_dir, f'adata_{i}.h5ad')
    # Save each AnnData object as h5ad
    adata.write(file_path)

# Make scanpy objects, do analysis in python

adata_obj_list = []
for i, class_array in enumerate(shap_values):
    print(i)
    df_class_array = pd.DataFrame(class_array)
    df_class_array.columns = dataset_test.columns
    df_class_array.index = dataset_test.index
    adata = sc.AnnData(df_class_array)
#    adata.X = csr_matrix(adata.X)
    # Append to list
    adata_obj_list.append(adata)
del adata, class_array, dataset_test, df_class_array, i, shap_values

# Create concatenate adata object
adata_com = ad.concat(adata_obj_list, label='Class', keys=np.arange(0, 13))  # concatenate list of anndata
del adata_obj_list
# Enforce unique barcode names
adata_com.obs_names_make_unique()
# adata_com.write('/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/shap_analysis/anndata_objects/background_100/adata_combined_background_100.h5ad')

#FROM SERVER
# adata_com = sc.read_h5ad('/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/shap_analysis/anndata_objects/background_1000/adata_combined_background_100.h5ad')

#FROM LOCAL STORAGE
# adata_com = sc.read_h5ad('/media/Storage/anndata_shap/background_1000/adata_combined.h5ad')

# ADD information about species class disease to anndata object
test_ground_truth = '/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/240830_SHAP_set_balanced_200_per_celltype_labels_binarized.csv'
test_ground_truth = pd.read_csv(test_ground_truth, header=None, low_memory=False)
test_labels = ('Human', 'Mice',
               'Cardiomyocytes', 'Endothelial', 'Fibroblasts', 'Immune.cells', 'Neuro', 'Pericytes', 'Smooth.Muscle',
               'AS', 'HFpEF', 'HFrEF', 'CTRL')
test_ground_truth.columns = test_labels
# Ask for 1s, replace with colnames
for col in test_ground_truth.columns:
    print(col)
    test_ground_truth[col] = test_ground_truth[col].apply(lambda x: col if x == 1 else 0)

# Same length as in anndata object, we have 13 classes with all cells, therefore we need 13 times the ground truth info
test_ground_truth_collect = [test_ground_truth]
for _ in range(12):
    test_ground_truth_collect.append(test_ground_truth)
test_ground_truth_collect = pd.concat(test_ground_truth_collect, ignore_index=True)

test_ground_truth_collect.index = adata_com.obs_names


# Concatenate strings
def concatenate_non_zero(row):
    return '_'.join(str(val) for val in row if val != 0)


test_ground_truth_collect['Concat_string'] = test_ground_truth_collect.apply(concatenate_non_zero, axis=1)
test_ground_truth_collect[['Species', 'Cell_type', 'Disease']] = test_ground_truth_collect['Concat_string'].str.split(
    '_', n=2, expand=True)

# Add to Anndata obs groups
adata_com.obs['Species'] = test_ground_truth_collect['Species']
adata_com.obs['Cell_type'] = test_ground_truth_collect['Cell_type']
adata_com.obs['Disease'] = test_ground_truth_collect['Disease']
adata_com.obs['Concat_string'] = test_ground_truth_collect['Concat_string']

# adata_com.write('/media/Storage/anndata_shap/background_200_balanced_data/adata_combined.h5ad')
# adata_com = sc.read_h5ad('/media/Storage/anndata_shap/background_200_balanced_data/adata_combined.h5ad')

'''
# Grab one of the shap value matrices and define the Differential predicted genes with groups
'''

#use z-scored shap values
adata_com = z_score_loop_per_cell(adata_com, sparse=False)
# adata_com.write('/media/Storage/anndata_shap/Z_SHAP/background_200_balanced_data/object/Z_adata_combined.h5ad')
# adata_com = sc.read_h5ad('/media/Storage/anndata_shap/Z_SHAP/background_200_balanced_data/object/Z_adata_combined.h5ad')

# Expression values
testpath = '/media/Storage/anndata_shap/whole_matrix_Seurat_extract/240830_SHAP_set_balanced_200_per_celltype.csv'
dataset_test = pd.read_csv(testpath, skiprows=0, header=0, index_col=0, low_memory=False)
# make a for loop for HUMAN relevant comparisons
pattern_1=[1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # start pattern 1 at Human CM AS
pattern_2=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # start pattern 2  at Human CM CTRL
shap_class=9 # Start at shap values from AS

#adjust thresholds, 0 for GSEA, otherwise use defaults
for i in range(7*3):
    simpleclass(adata_com,
                shap_class=shap_class,
                pattern_1=pattern_1,
                pattern_2=pattern_2,
                testing_method="t", # write either t or wilcoxon
                sparse_matrix=False,
                SHAP_thresh=0.01,
                logfc_thresh=0.1,
                expression_threshold_test_data=0.1,
                p_adj_thresh=0.05,
                ground_truth='/media/Storage/anndata_shap/whole_matrix_Seurat_extract/HUMAN_241121_SHAP_set_balanced_200_binarized_labels.csv',
                ground_truth_exp=dataset_test,
                outDir='/media/Storage/anndata_shap/Z_SHAP/background_200_Human_balanced_data')
    pattern_1, pattern_2 = next_pattern(pattern_1, pattern_2)
    shap_class +=1
    if shap_class > 11:
        shap_class = 9

# make a for loop for Mouse relevant comparisons
pattern_1=[0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # start pattern 1 at Mouse CM AS
pattern_2=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # start pattern 2  at Mouse CM CTRL
shap_class=9 # Start at shap values from AS
for i in range(7*3):
    simpleclass(adata_com,
                shap_class=shap_class,
                pattern_1=pattern_1,
                pattern_2=pattern_2,
                testing_method="wilcoxon", # write either t or wilcoxon
                sparse_matrix=False,
                SHAP_thresh = 0.01,
                logfc_thresh = 0.1,
                expression_threshold_test_data=1,
                p_adj_thresh=0.05,
                ground_truth='/media/Helios_scStorage/Mariano/NN_Human_Mice/f1_loss_runs_24_06_18/240830_SHAP_set_balanced_200_per_celltype_labels_binarized.csv',
                ground_truth_exp=dataset_test,
                outDir='/media/Storage/anndata_shap/Z_SHAP/background_200_balanced_data')

    pattern_1, pattern_2 = next_pattern(pattern_1, pattern_2)
    shap_class +=1
    if shap_class > 11:
        shap_class = 9







