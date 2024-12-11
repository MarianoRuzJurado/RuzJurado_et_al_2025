import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the 'Agg' backend which is non-interactive and doesn't require a display

outDir = '/media/Helios_scStorage/Mariano/NN_Human_Mice/Paper_R_code/Dotplots'

# Load adata for sc.pl.dotplot
adata_obj_H = sc.read_h5ad(
    "/media/Helios_scStorage/Mariano/NN_Human_Mice/Paper_R_code/objects/adata.Obj.annotated.H.h5ad")
adata_obj_M = sc.read_h5ad(
    "/media/Helios_scStorage/Mariano/NN_Human_Mice/Paper_R_code/objects/adata.Obj.annotated.M.h5ad")

'''
Human_tp5_markers = pd.read_excel(
    "/media/Helios_scStorage/Mariano/NN_Human_Mice/Paper_R_code/excelsheets/markers_celtype/Human_tp5_markers_cell_annotation.xlsx",
    index_col=0)
Mouse_tp5_markers = pd.read_excel(
    "/media/Helios_scStorage/Mariano/NN_Human_Mice/Paper_R_code/excelsheets/markers_celtype/Mouse_tp5_markers_cell_annotation.xlsx",
    index_col=0)
'''

dict_cell_Markers_h = {'CM': ['RYR2', 'MLIP', 'TTN', 'FGF12', 'FHL2'],
                     'EC': ['VWF', 'ANO2', 'PTPRB', 'LDB2', 'FLT1'],
                     'FB': ['ABCA6', 'PDGFRA', 'DCN', 'MGP', 'ABCA9'],
                     'IC': ['F13A1', 'RBPJ', 'MSR1', 'CD163', 'MRC1'],
                     'NC': ['NRXN1', 'XKR4', 'CHL1', 'NRXN3', 'CADM2'],
                     'PC': ['RGS5', 'GUCY1A2', 'EGFLAM', 'ABCC9', 'FRMD3'],
                    'SMC': ['MYH11', 'ITGA8', 'NTRK3', 'ACTA2','TAGLN'],
}  # Markers from Hocker et al 2021, Koenig et al 2022, Litvinukova et al 2020
dp = sc.pl.dotplot(adata_obj_H,
                   var_names=dict_cell_Markers_h,
                   groupby="cell_type",
                   log=True,
                   show=False,
                   swap_axes=True,
                   figsize=(5,8.5),
                   categories_order=sorted(adata_obj_H.obs['cell_type'].unique()))

# All Axes used in dotplot
print("Dotplot axes:", dp)
# Select the Axes object that contains the subplot of interest
ax = dp["mainplot_ax"]

# Remove the x-axis labels (groupby categories) from the bottom
lab = ax.get_xticklabels()
# ax.set_xticks([])

# Loop through ticklabels and make them italic
for l in ax.get_yticklabels():
    l.set_style("italic")
    l.set_weight("bold")

for l in ax.get_xticklabels():
    l.set_weight("bold")

for txt in dp['gene_group_ax'].texts:
    txt.set_weight('bold')

plt.savefig(f'{outDir}/Dotplot_Human_tp5_markers.pdf', dpi=300)

#with rotated axis
dp = sc.pl.dotplot(adata_obj_H,
                   var_names=dict_cell_Markers_h,
                   groupby="cell_type",
                   log=True,
                   show=False,
                   swap_axes=False,
                   figsize=(15,4),
                   categories_order=sorted(adata_obj_H.obs['cell_type'].unique()))

# All Axes used in dotplot
print("Dotplot axes:", dp)
# Select the Axes object that contains the subplot of interest
ax = dp["mainplot_ax"]

# Remove the x-axis labels (groupby categories) from the bottom
lab = ax.get_yticklabels()
# ax.set_yticks([])

# Loop through ticklabels and make them italic
for l in ax.get_xticklabels():
    l.set_style("italic")
    l.set_weight("bold")
    l.set_fontsize(14)

for l in ax.get_yticklabels():
    l.set_weight("bold")
    l.set_fontsize(14)

for txt in dp['gene_group_ax'].texts:
    txt.set_weight('bold')
    txt.set_fontsize(14)

plt.savefig(f'{outDir}/Dotplot_Human_tp5_markers_rotated.pdf', dpi = 300)

# Mouse
dict_cell_Markers_m = {'CM': ['RYR2', 'MLIP', 'TTN', 'RBM20', 'FHL2'],
                     'EC': ['VWF', 'MECOM', 'PTPRB', 'LDB2', 'FLT1'],
                     'FB': ['ABCA6', 'PDGFRA', 'DCN', 'MGP', 'ABCA9'],
                     'IC': ['F13A1', 'RBPJ', 'MSR1', 'CD163', 'MRC1'],
                     'NC': ['NRXN1', 'XKR4', 'CHL1', 'NRXN3', 'CADM2'],
                     'PC': ['RGS5', 'GUCY1A2', 'EGFLAM', 'ABCC9', 'FRMD3'],
                    'SMC': ['MYH11', 'ITGA8', 'NTRK3', 'ACTA2','TAGLN'],
}

dp = sc.pl.dotplot(adata_obj_M,
                   var_names=dict_cell_Markers_m,
                   groupby="cell_type",
                   log=True,
                   show=False,
                   swap_axes=True,
                   figsize=(5,8.5),
                   categories_order=sorted(adata_obj_M.obs['cell_type'].unique()))


# All Axes used in dotplot
print("Dotplot axes:", dp)
# Select the Axes object that contains the subplot of interest
ax = dp["mainplot_ax"]

# Remove the x-axis labels (groupby categories) from the bottom
lab = ax.get_xticklabels()
# ax.set_xticks([])

# Loop through ticklabels and make them italic
for l in ax.get_yticklabels():
    l.set_style("italic")
    l.set_weight("bold")

for l in ax.get_xticklabels():
    l.set_weight("bold")

for txt in dp['gene_group_ax'].texts:
    txt.set_weight('bold')

plt.savefig(f'{outDir}/Dotplot_Mouse_tp5_markers.pdf', dpi = 300)

dp = sc.pl.dotplot(adata_obj_M,
                   var_names=dict_cell_Markers_m,
                   groupby="cell_type",
                   log=True,
                   show=False,
                   swap_axes=False,
                   figsize=(15,4),
                   categories_order=sorted(adata_obj_M.obs['cell_type'].unique()))


# All Axes used in dotplot
print("Dotplot axes:", dp)
# Select the Axes object that contains the subplot of interest
ax = dp["mainplot_ax"]

# Remove the x-axis labels (groupby categories) from the bottom
lab = ax.get_xticklabels()
# ax.set_xticks([])

# Loop through ticklabels and make them italic
for l in ax.get_xticklabels():
    l.set_style("italic")
    l.set_weight("bold")
    l.set_fontsize(14)

for l in ax.get_yticklabels():
    l.set_weight("bold")
    l.set_fontsize(14)

for txt in dp['gene_group_ax'].texts:
    txt.set_weight('bold')
    txt.set_fontsize(14)

plt.savefig(f'{outDir}/Dotplot_Mouse_tp5_markers_rotated.pdf', dpi = 300)




# Set the x-axis label to display groupby categories at the top
# cat = adata_obj_H.obs["cell_type"].unique()
# cat = ', '.join(cat)
# ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
# ax.xaxis.set_label_position('top')  # Set the label position to top
# ax.set_xlabel(lab, fontsize=12, rotation=90)
