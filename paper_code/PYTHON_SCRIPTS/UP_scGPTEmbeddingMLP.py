import argparse, pickle
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from UP_preprocess import MultiLabelBin, createDatasets
from UP_models import MLP_Model
from UP_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


trainpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_train_set.pkl'
valpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_val_set.pkl'
trainlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_train_set_rawCounts_labels_p_80.csv'
vallabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/RANDOMIZED_val_set_rawCounts_labels_p_20.csv'
testpath = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_test_set.pkl'
testlabels = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/objects/test_set_rawCounts_labels.csv'
batchsize = 10240
seed = 1234
tf.random.set_seed(1234)
MLPLayers = 3
MLPNeurons = [795, 230, 105]
outputFolder = '/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/RUN8'

#Loading embedded data from scGPT
dataset_train = pickle.load(
    open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_train_set.pkl',
        'rb'))
dataset_val = pickle.load(
    open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_val_set.pkl',
        'rb'))
dataset_test = pickle.load(
    open('/mnt/mariano/Sequencing_Storage/scStorage/Mariano/NN_Human_Mice/PaperExternRevision/250812_scGPT/embeddings_extr/embedding_test_set.pkl',
        'rb'))

# Do the multi-label-hot encoding
train_labels_bin = MultiLabelBin(trainlabels)
val_labels_bin = MultiLabelBin(vallabels)
test_labels_bin = MultiLabelBin(testlabels)



# Set Save string for MLP
MLP_neurons_str = '_'.join(map(str, MLPNeurons))

np.savetxt(outputFolder + '/test_labels_binarized_' + 'MLP_' + MLP_neurons_str + '_Batch_' + str(batchsize) + '_SEED_' + str(seed) + '.csv',
               test_labels_bin, delimiter=',')

input_data = dataset_train
input_shape = dataset_train.shape[1]
input_val = dataset_val

early_stop = EarlyStopping(monitor='val_loss', patience=50)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, min_lr=1e-7)  # LR reduction ADAM callback

mlp_model_instance = MLP_Model(input_shape, train_labels_bin.shape[1], MLPLayers, MLPNeurons)
mlp_model_instance.compile(optimizer='adam', loss=macro_f1_loss,
                           metrics=[f1_mean, f1_species, f1_celltype, f1_disease,
                                    precision_mean, precision_species, precision_celltype, precision_disease,
                                    recall_mean, recall_species, recall_celltype, recall_disease,
                                    AUC(name='roc_auc', curve='ROC', multi_label=True, num_labels=13),
                                    AUC(name='pr_auc', curve='PR', multi_label=True, num_labels=13)], )
print(get_time() + "Start fitting MLP...", )
history_mlp = mlp_model_instance.fit(input_data, train_labels_bin, epochs=1500, batch_size=batchsize,
                                             validation_data=(input_val, val_labels_bin),
                                             callbacks=[reduce_lr, early_stop])
# Add epoch numbers
history_mlp.history['epoch'] = list(range(1, len(history_mlp.history['loss']) + 1))

with open(
        f'{outputFolder}/MLP_Training_history_{MLP_neurons_str}_Batch_{batchsize}_SEED_{seed}.pkl',
        'wb') as file_pi:
    pickle.dump(history_mlp.history, file_pi)

print(get_time() + "Training finished for MLP...", )

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['loss'],
           history_mlp.history['val_loss'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_Train_val_loss",
           ylim=[0.00, 0.7])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['f1_mean'],
           history_mlp.history['val_f1_mean'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_F1_train_val_mean",
           ylim=[0.7, 1.005])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['f1_species'],
           history_mlp.history['val_f1_species'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_F1_train_val_species",
           ylim=[0.8, 1.005])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['f1_celltype'],
           history_mlp.history['val_f1_celltype'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_F1_train_val_celltype",
           ylim=[0.7, 1.005])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['f1_disease'],
           history_mlp.history['val_f1_disease'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_F1_train_val_disease",
           ylim=[0.3, 1.005])

loss_curve_prec_rec(history_mlp.history['epoch'],
                    history_mlp.history['precision_mean'],
                    history_mlp.history['val_precision_mean'],
                    history_mlp.history['recall_mean'],
                    history_mlp.history['val_recall_mean'],
                    metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
                    save_dir=f'{outputFolder}/Metric_figures',
                    name="MLP_prec_rec_train_val_mean",
                    ylim=[0.7, 1.005])

loss_curve_prec_rec(history_mlp.history['epoch'],
                    history_mlp.history['precision_species'],
                    history_mlp.history['val_precision_species'],
                    history_mlp.history['recall_species'],
                    history_mlp.history['val_recall_species'],
                    metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
                    save_dir=f'{outputFolder}/Metric_figures',
                    name="MLP_prec_rec_train_val_species",
                    ylim=[0.8, 1.005])

loss_curve_prec_rec(history_mlp.history['epoch'],
                    history_mlp.history['precision_celltype'],
                    history_mlp.history['val_precision_celltype'],
                    history_mlp.history['recall_celltype'],
                    history_mlp.history['val_recall_celltype'],
                    metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
                    save_dir=f'{outputFolder}/Metric_figures',
                    name="MLP_prec_rec_train_val_celltype",
                    ylim=[0.7, 1.005])

loss_curve_prec_rec(history_mlp.history['epoch'],
                    history_mlp.history['precision_disease'],
                    history_mlp.history['val_precision_disease'],
                    history_mlp.history['recall_disease'],
                    history_mlp.history['val_recall_disease'],
                    metrics_name=['prec_train', 'prec_val', 'rec_train', 'rec_val'],
                    save_dir=f'{outputFolder}/Metric_figures',
                    name="MLP_prec_rec_train_val_disease",
                    ylim=[0.3, 1.005])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['roc_auc'],
           history_mlp.history['val_roc_auc'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_Train_val_roc_auc",
           ylim=[0.6, 1.005])

loss_curve(history_mlp.history['epoch'],
           history_mlp.history['pr_auc'],
           history_mlp.history['val_pr_auc'],
           save_dir=f'{outputFolder}/Metric_figures',
           name="MLP_Train_val_pr_auc",
           ylim=[0.6, 1.005])

mlp_model_instance.save(
    outputFolder + '/MLP_' + MLP_neurons_str + '_Batch_' + str(batchsize) + '_SEED_' + str(
        seed) + '.keras')
print(get_time() + "MLP Saved!")

print(get_time() + "Testing Phase: Predict with MLP...", )
pred_Test = mlp_model_instance.predict(dataset_test)
print(get_time() + "Testing Phase: Prediction with MLP finished!", )

np.savetxt(outputFolder  + '/Pred_raw_results' + '_MLP_' + MLP_neurons_str + '_Batch_' + str(batchsize) + '_SEED_' + str(seed) + '.csv', pred_Test, delimiter=',')
pred_result = onehot_predict(pred_Test)
np.savetxt(outputFolder + '/Pred_results' + '_MLP_' + MLP_neurons_str + '_Batch_' + str(batchsize) + '_SEED_' + str(seed) + '.csv', pred_result, delimiter=',')


