import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib.pyplot as plt
from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig, scGPT
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

trainpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_train_set_rawCounts_p_80.csv'
valpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_val_set_rawCounts_p_20.csv'
trainlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_train_set_rawCounts_labels_p_80.csv'
vallabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_val_set_rawCounts_labels_p_20.csv'
testpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/test_set_rawCounts.csv'
testlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/test_set_rawCounts_labels.csv'

# Read data
dataset_train = pd.read_csv(trainpath, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_trainlabels = pd.read_csv(trainlabels, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_val = pd.read_csv(valpath, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_vallabels = pd.read_csv(vallabels, skiprows=0, header=0, index_col=0, low_memory=False)

# Read test
dataset_test = pd.read_csv(testpath, skiprows=0, header=0, index_col=0, low_memory=False)
dataset_testlabels = pd.read_csv(testlabels, skiprows=0, header=0, index_col=0, low_memory=False)


# Make anndata for structure provided in helicalAi
adata_train = ad.AnnData(dataset_train)
adata_train.X = sp.csr_matrix(adata_train.X)
adata_val = ad.AnnData(dataset_val)
adata_val.X = sp.csr_matrix(adata_val.X)

# Make anndata test
adata_test = ad.AnnData(dataset_test)
adata_test.X = sp.csr_matrix(adata_test.X)

# Populate the obs with metadata TRAIN
dataset_trainlabels = dataset_trainlabels.reset_index().iloc[:,0].str.split('_|-', n=2, expand=True)
dataset_trainlabels.columns = ['species', 'condition', 'celltype']
adata_train.obs[dataset_trainlabels.columns] = dataset_trainlabels.values
adata_train.obs['concat_string'] =  adata_train.obs['species'].astype(str) + '_' +  adata_train.obs['condition'].astype(str) + '_' +  adata_train.obs['celltype'].astype(str)

adata_train.write('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_train.h5ad')
# adata_train = sc.read_h5ad('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_train.h5ad')
# OnSERVER adata_train = sc.read_h5ad('/home/mruz/NN_Human_Mice/250812_scGPT/objects/adata_train.h5ad')

# Populate the obs with metadata VAL
dataset_vallabels = dataset_vallabels.reset_index().iloc[:,0].str.split('_|-', n=2, expand=True)
dataset_vallabels.columns = ['species', 'condition', 'celltype']
adata_val.obs[dataset_vallabels.columns] = dataset_vallabels.values
adata_val.obs['concat_string'] =  adata_val.obs['species'].astype(str) + '_' +  adata_val.obs['condition'].astype(str) + '_' +  adata_val.obs['celltype'].astype(str)

adata_val.write('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_val.h5ad')
# adata_val = sc.read_h5ad('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_val.h5ad')
# OnSERVER adata_val = sc.read_h5ad('/home/mruz/NN_Human_Mice/250812_scGPT/objects/adata_val.h5ad')

# Populate the obs with metadata TEST
dataset_testlabels = dataset_testlabels.reset_index().iloc[:,0].str.split('_|-', n=2, expand=True)
dataset_testlabels.columns = ['species', 'condition', 'celltype']
adata_test.obs[dataset_testlabels.columns] = dataset_testlabels.values
adata_test.obs['concat_string'] =  adata_test.obs['species'].astype(str) + '_' +  adata_test.obs['condition'].astype(str) + '_' +  adata_test.obs['celltype'].astype(str)

adata_test.write('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_test.h5ad')
# adata_test = sc.read_h5ad('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/adata_test.h5ad')
# OnSERVER adata_test = sc.read_h5ad('/home/mruz/NN_Human_Mice/250812_scGPT/objects/adata_test.h5ad')

# Follow instruction scGPT CELLTYPE

cell_types_train = list(adata_train.obs.celltype)
label_set_train = set(cell_types_train)

cell_types_val = list(adata_val.obs.celltype)
label_set_val = set(cell_types_val)

cell_types_test = list(adata_test.obs.celltype)
label_set_test = set(cell_types_test)

scgpt_config = scGPTConfig(batch_size=10, device="cuda")
scgpt_fine_tune = scGPTFineTuningModel(scGPT_config=scgpt_config, fine_tuning_head='classification', output_size=len(label_set_train))

data_train = scgpt_fine_tune.process_data(adata_train)
data_val = scgpt_fine_tune.process_data(adata_val)
data_test = scgpt_fine_tune.process_data(adata_test)

class_id_dict = dict(zip(label_set_train, [i for i in range(len(label_set_train))]))

for i in range(len(cell_types_train)):
    cell_types_train[i] = class_id_dict[cell_types_train[i]]

for i in range(len(cell_types_val)):
    cell_types_val[i] = class_id_dict[cell_types_val[i]]

for i in range(len(cell_types_test)):
    cell_types_test[i] = class_id_dict[cell_types_test[i]]

scgpt_fine_tune.train(train_input_data=data_train,
                      train_labels=cell_types_train,
                      validation_input_data=data_val,
                      validation_labels=cell_types_val,
                      epochs=1)


outputs_test = scgpt_fine_tune.get_outputs(data_test)

print(classification_report(cell_types_test, outputs_test.argmax(axis=1)))

# Compute the confusion matrix
cm = confusion_matrix(cell_types_test, outputs_test.argmax(axis=1))

# Perform row-wise normalization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Get unique labels in the order they appear in the confusion matrix
unique_labels = np.unique(np.concatenate((cell_types_test, outputs_test.argmax(axis=1))))

# Use id_class_dict to get the class names
id_to_class = {v: k for k, v in class_id_dict.items()}
class_names = [class_id_dict[label] for label in unique_labels]

# Create and plot the normalized confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation='vertical', values_format='.2f', cmap='Reds')

for text in disp.text_.ravel():
    text.set_fontsize(14)   # adjust this number as you like

# Customize the plot
ax.set_title('Normalized Confusion Matrix (Row-wise)')
fig.set_facecolor("none")

plt.tight_layout()
plt.show()
plt.savefig('/home/mruz/NN_Human_Mice/250812_scGPT/results/scGPT_cm_celltype.pdf')

####

# Follow instruction scGPT SPECIES

species_train = list(adata_train.obs.species)
label_set_train = set(species_train)

species_val = list(adata_val.obs.species)
label_set_val = set(species_val)

species_test = list(adata_test.obs.species)
label_set_test = set(species_test)

scgpt_config = scGPTConfig(batch_size=10, device="cuda")
scgpt_fine_tune = scGPTFineTuningModel(scGPT_config=scgpt_config, fine_tuning_head='classification', output_size=len(label_set_train))

data_train = scgpt_fine_tune.process_data(adata_train)
data_val = scgpt_fine_tune.process_data(adata_val)
data_test = scgpt_fine_tune.process_data(adata_test)

class_id_dict = dict(zip(label_set_train, [i for i in range(len(label_set_train))]))

for i in range(len(species_train)):
    species_train[i] = class_id_dict[species_train[i]]

for i in range(len(species_val)):
    species_val[i] = class_id_dict[species_val[i]]

for i in range(len(species_test)):
    species_test[i] = class_id_dict[species_test[i]]

scgpt_fine_tune.train(train_input_data=data_train,
                      train_labels=species_train,
                      validation_input_data=data_val,
                      validation_labels=species_val,
                      epochs=10)


outputs_test = scgpt_fine_tune.get_outputs(data_test)

print(classification_report(species_test, outputs_test.argmax(axis=1)))

# Compute the confusion matrix
cm = confusion_matrix(species_test, outputs_test.argmax(axis=1))

# Perform row-wise normalization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Get unique labels in the order they appear in the confusion matrix
unique_labels = np.unique(np.concatenate((species_test, outputs_test.argmax(axis=1))))

# Use id_class_dict to get the class names
id_to_class = {v: k for k, v in class_id_dict.items()}
class_names = [class_id_dict[label] for label in unique_labels]

# Create and plot the normalized confusion matrix
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation='vertical', values_format='.2f', cmap='Reds')

for text in disp.text_.ravel():
    text.set_fontsize(14)   # adjust this number as you like

# Customize the plot
ax.set_title('Normalized Confusion Matrix (Row-wise)')
fig.set_facecolor("none")

plt.tight_layout()
plt.show()
plt.savefig('/home/mruz/NN_Human_Mice/250812_scGPT/results/scGPT_cm_species.pdf')

####

# Follow instruction scGPT CONDITION

condition_train = list(adata_train.obs.condition)
label_set_train = set(condition_train)

condition_val = list(adata_val.obs.condition)
label_set_val = set(condition_val)

condition_test = list(adata_test.obs.condition)
label_set_test = set(condition_test)

scgpt_config = scGPTConfig(batch_size=10, device="cuda")
scgpt_fine_tune = scGPTFineTuningModel(scGPT_config=scgpt_config, fine_tuning_head='classification', output_size=len(label_set_train))

data_train = scgpt_fine_tune.process_data(adata_train)
data_val = scgpt_fine_tune.process_data(adata_val)
data_test = scgpt_fine_tune.process_data(adata_test)

class_id_dict = dict(zip(label_set_train, [i for i in range(len(label_set_train))]))

for i in range(len(condition_train)):
    condition_train[i] = class_id_dict[condition_train[i]]

for i in range(len(condition_val)):
    condition_val[i] = class_id_dict[condition_val[i]]

for i in range(len(condition_test)):
    condition_test[i] = class_id_dict[condition_test[i]]

scgpt_fine_tune.train(train_input_data=data_train,
                      train_labels=condition_train,
                      validation_input_data=data_val,
                      validation_labels=condition_val,
                      epochs=1)


outputs_test = scgpt_fine_tune.get_outputs(data_test)

print(classification_report(condition_test, outputs_test.argmax(axis=1)))

# Compute the confusion matrix
cm = confusion_matrix(condition_test, outputs_test.argmax(axis=1))

# Perform row-wise normalization
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Get unique labels in the order they appear in the confusion matrix
unique_labels = np.unique(np.concatenate((condition_test, outputs_test.argmax(axis=1))))

# Use id_class_dict to get the class names
id_to_class = {v: k for k, v in class_id_dict.items()}
class_names = [class_id_dict[label] for label in unique_labels]

# Create and plot the normalized confusion matrix
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation='vertical', values_format='.2f', cmap='Reds')

for text in disp.text_.ravel():
    text.set_fontsize(14)   # adjust this number as you like

# Customize the plot
ax.set_title('Normalized Confusion Matrix (Row-wise)')
fig.set_facecolor("none")

plt.tight_layout()
plt.show()
plt.savefig('/home/mruz/NN_Human_Mice/250812_scGPT/results/scGPT_cm_disease.pdf')

np.savetxt("/home/mruz/NN_Human_Mice/250812_scGPT/results/scGPT_species_pred_raw_result.csv", outputs_test, delimiter=",")
save_path = "/home/mruz/NN_Human_Mice/250812_scGPT/models/scGPT_species_tuned.pth"
import torch
torch.save(scgpt_fine_tune.model.state_dict(), save_path)
save_path = "/home/mruz/NN_Human_Mice/250812_scGPT/models/scGPT_species_tuned_full.pth"
torch.save(scgpt_fine_tune.model, save_path)
import pickle
with open('/home/mruz/NN_Human_Mice/250812_scGPT/results/output_test_species.pkl', 'wb') as file_pi:
    pickle.dump(outputs_test, file_pi)


##### get embedded version of data and train an MLP on that embedding
scgpt_config = scGPTConfig(batch_size=312, device="cuda")
scgpt = scGPT(configurer = scgpt_config)

data_train = scgpt.process_data(adata_train)
data_val = scgpt.process_data(adata_val)
data_test = scgpt.process_data(adata_test)

# emb + save train data
data_train_emb = scgpt.get_embeddings(data_train)
with open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_train_set.pkl', 'wb') as file_pi:
    pickle.dump(data_train_emb, file_pi)

#emb + save val data
data_val_emb = scgpt.get_embeddings(data_val)
with open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_val_set.pkl', 'wb') as file_pi:
    pickle.dump(data_val_emb, file_pi)

# emb + save test data
data_test_emb = scgpt.get_embeddings(data_test)
with open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_test_set.pkl', 'wb') as file_pi:
    pickle.dump(data_test_emb, file_pi)
