import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.ticker import MaxNLocator

# Path to log files
log_file_path_1 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/240517_500GiB_Trials_4000_6000_2000_3000_300_500_5_5_5/NN_AE_HYPERTUNING_SCRIPT.log'
log_file_path_2 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/240525_500Gib_Trials_4800_5600_2000_2500_300_500_50_50_50/NN_AE_HYPERTUNING_SCRIPT.log'
log_file_path_3 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/240528_500Gib_Trials_5000_5300_2200_2500_300_400_50_50_50/NN_AE_HYPERTUNING_SCRIPT.log'
log_file_path_4 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/AE_240607.log'
log_file_path_5 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/AE_240612.log'
# Regular expression to match the loss value after each trial completion
loss_pattern = re.compile(r"Trial \d+ Complete \[\d+h \d+m \d+s\]\nloss: ([0-9.]+)")
# Regular expression to match the values under the "Value" column for each trial
neuron_pattern = re.compile(r"^\d{1,5}(?=\s+\|)", re.MULTILINE)

# List to store the loss values
loss_values_1 = []

with open(log_file_path_1, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_1 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_1):
    print(f"Trial {idx + 1}: Loss = {loss}")


# List to store the averages of the first six values
averages_1 = []
averages_1_enc = []
averages_1_dec = []

with open(log_file_path_1, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:5]]  # Get first 5 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_1.append(average_value)
            averages_1_enc.append(sum([values[i] for i in [0, 2, 4]]) / 3)
            averages_1_dec.append(sum([values[i] for i in [1, 3]]) / 2)

# List to store the loss values
loss_values_2 = []

with open(log_file_path_2, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_2 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_2):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_2 = []
averages_2_enc = []
averages_2_dec = []

with open(log_file_path_2, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:5]]  # Get first 5 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_2.append(average_value)
            averages_2_enc.append(sum([values[i] for i in [0, 2, 4]]) / 3)
            averages_2_dec.append(sum([values[i] for i in [1, 3]]) / 2)

# List to store the loss values
loss_values_3 = []

with open(log_file_path_3, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_3 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_3):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_3 = []
averages_3_enc = []
averages_3_dec = []

with open(log_file_path_3, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:5]]  # Get first 5 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_3.append(average_value)
            averages_3_enc.append(sum([values[i] for i in [0, 2, 4]]) / 3)
            averages_3_dec.append(sum([values[i] for i in [1, 3]]) / 2)

# List to store the loss values
loss_values_4 = []

with open(log_file_path_4, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_4 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_4):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_4 = []
averages_4_enc = []
averages_4_dec = []

with open(log_file_path_4, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:5]]  # Get first 5 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_4.append(average_value)
            averages_4_enc.append(sum([values[i] for i in [0, 2, 4]]) / 3)
            averages_4_dec.append(sum([values[i] for i in [1, 3]]) / 2)

# List to store the loss values
loss_values_5 = []

with open(log_file_path_5, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_5 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_5):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_5 = []
averages_5_enc = []
averages_5_dec = []

with open(log_file_path_5, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:5]]  # Get first 5 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_5.append(average_value)
            averages_5_enc.append(sum([values[i] for i in [0, 1, 2]]) / 3)
            averages_5_dec.append(sum([values[i] for i in [3, 4]]) / 2)


loss_values_comb = loss_values_1 + loss_values_2 + loss_values_3 + loss_values_4 + loss_values_5
averages_comb = averages_1 + averages_2 + averages_3 +averages_4 + averages_5
averages_comb_enc = averages_1_enc + averages_2_enc + averages_3_enc + averages_4_enc + averages_5_enc
averages_comb_dec = averages_1_dec + averages_2_dec + averages_3_dec + averages_4_dec + averages_5_dec

# Some runs at the end didn't have loss calculations so cut the average variable by the length of loss values
averages_comb = averages_comb[:len(loss_values_comb)]
averages_comb_enc = averages_comb_enc[:len(loss_values_comb)]
averages_comb_dec = averages_comb_dec[:len(loss_values_comb)]

# Define custom colormap
colors = [(128/255, 0, 0), (47/255, 79/255, 79/255)]  # Maroon, white, dark slate gray
custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors)

# Create a 3D plot
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

# Change the color of the axis "box"
ax.set_box_aspect([1,1,1])

# Change the color of the axis "box"
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

# Normalize loss values to range [0, 1] for colormap
norm = plt.Normalize(min(loss_values_comb), max(loss_values_comb))

# Scatter plot
scatter = ax.scatter(loss_values_comb,
           averages_comb_enc,
           averages_comb_dec,
           c=loss_values_comb,
           cmap=custom_cmap,
           norm=norm)
# Set the number of ticks on the y-axis to 5
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

# Add color bar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Loss')

# Find index of minimum loss value
min_loss_index = np.argmin(loss_values_comb)

# Plot point with minimum loss value differently
ax.scatter(loss_values_comb[min_loss_index],
                     averages_comb_enc[min_loss_index],
                     averages_comb_dec[min_loss_index],
                     c='green',
                     s=100,
                     label='Min Loss')
ax.invert_xaxis()



# Labeling axes
ax.set_ylabel('Average Neurons (Encoder)')
ax.set_zlabel('Average Neurons (Decoder)')
ax.set_xlabel('L2-Loss')

# Title
ax.set_title('3D Scatter Plot of Loss vs Average Neuron Values')
#ax.view_init(0,180)
plt.show()
plt.savefig("/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/plot/AE_3d_hypertuning_plot_test.pdf")
plt.close()


#MLP_hypertuning_analysis

# Path to log files
log_file_path_1 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/MLP_240619_2.log'
log_file_path_2 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/MLP_240619_2_2.log'
log_file_path_3 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/MLP_240618_3.log'
log_file_path_4 = '/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/local_run/MLP_240618_4.log'

# Regular expression to match the loss value after each trial completion
loss_pattern = re.compile(r"Trial \d+ Complete \[\d+h \d+m \d+s\]\nval_macro_f1_loss: ([0-9.]+)")
# Regular expression to match the values under the "Value" column for each trial
neuron_pattern = re.compile(r"^\d{1,5}(?=\s+\|)", re.MULTILINE)

# List to store the loss values
loss_values_1 = []

with open(log_file_path_1, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_1 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_1):
    print(f"Trial {idx + 1}: Loss = {loss}")


# List to store the averages of the first six values
averages_1 = []

with open(log_file_path_1, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:2]]  # Get first 2 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_1.append(average_value)

averages_1 = averages_1[:len(loss_values_1)]
layers_1 = [2] * len(averages_1)

# List to store the loss values
loss_values_2 = []

with open(log_file_path_2, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_2 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_2):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_2 = []

with open(log_file_path_2, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:2]]  # Get first 2 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_2.append(average_value)

averages_2 = averages_2[:len(loss_values_2)]
layers_2 = [2] * len(averages_2)

# List to store the loss values
loss_values_3 = []

with open(log_file_path_3, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_3 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_3):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_3 = []

with open(log_file_path_3, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:2]]  # Get first 2 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_3.append(average_value)

averages_3 = averages_3[:len(loss_values_3)]
layers_3 = [3] * len(averages_3)

# List to store the loss values
loss_values_4= []

with open(log_file_path_4, 'r') as file:
    log_content = file.read()
    matches = loss_pattern.findall(log_content)
    loss_values_4 = [float(match) for match in matches]

# Print the extracted loss values
for idx, loss in enumerate(loss_values_4):
    print(f"Trial {idx + 1}: Loss = {loss}")

# List to store the averages of the first six values
averages_4 = []

with open(log_file_path_4, 'r') as file:
    log_content = file.read()
    # Find all matches for each trial
    trials = log_content.split("Search: Running Trial #")
    for trial in trials[1:]:  # Skip the first split part which is before the first trial
        values = [int(match) for match in neuron_pattern.findall(trial)[:2]]  # Get first 2 matches
        print(values)
        if values:
            average_value = sum(values) / len(values)
            averages_4.append(average_value)

averages_4 = averages_4[:len(loss_values_4)]
layers_4 = [4] * len(averages_4)

loss_values_comb = loss_values_1 + loss_values_2 + loss_values_3 + loss_values_4
loss_values_comb = loss_values_comb[:len(averages_comb)]
averages_comb = averages_1 + averages_2 + averages_3 + averages_4
layers_comb = layers_1 + layers_2 + layers_3 + layers_4

# Define custom colormap
colors = [(128/255, 0, 0), (47/255, 79/255, 79/255)]  # Maroon, white, dark slate gray
custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors)

# Create a 3D plot
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

# Change the color of the axis "box"
ax.set_box_aspect([1,1,1])

# Change the color of the axis "box"
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')

# Normalize loss values to range [0, 1] for colormap
norm = plt.Normalize(min(loss_values_comb), max(loss_values_comb))

# Scatter plot
scatter = ax.scatter(loss_values_comb,
           layers_comb,
           averages_comb,
           c=loss_values_comb,
           cmap=custom_cmap,
           norm=norm)
# Set the number of ticks on the y-axis to 5
ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

# Add color bar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Loss')

# Find index of minimum loss value
min_loss_index = np.argmin(loss_values_comb)

# Plot point with minimum loss value differently
ax.scatter(loss_values_comb[min_loss_index],
                     averages_comb_enc[min_loss_index],
                     averages_comb_dec[min_loss_index],
                     c='green',
                     s=100,
                     label='Min Loss')
ax.invert_xaxis()

# Labeling axes
ax.set_ylabel('Average Neurons')
ax.set_zlabel('Number of Layers')
ax.set_xlabel('F1-Loss')

# Title
ax.set_title('3D Scatter Plot of Loss vs Average Neuron Values')
#ax.view_init(0,180)
plt.show()
plt.savefig("/media/Helios_scStorage/Mariano/NN_Human_Mice/hypertuning/plot/MLP_3d_hypertuning_plot.pdf")
plt.close()
# Make a 2D plot out of it

# Define custom colormap
colors = [(128/255, 0, 0),
          (135/255, 0, 0),
          (150/255, 0, 0),
          (71/255, 121/255, 121/255),
          (62/255, 105/255, 105/255),
          (47/255, 79/255, 79/255)]
custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors)

#reorder layers and values for plot
loss_values_list = [loss_values_3, loss_values_1, loss_values_2, loss_values_5, loss_values_6, #loss_values_3 and _4 are both runs with 2 layers, keep the one with better result
                    loss_values_7, loss_values_8, loss_values_9, loss_values_10]

averages_comb_list = [averages_3, averages_1, averages_2, averages_5, averages_6,
                      averages_7, averages_8, averages_9, averages_10]

# Initialize a list to store the minimum loss values
loss_min_values = []
averages_comb_min = []

# Loop through the list of loss values lists
for loss_values in loss_values_list:
    min_loss_index = np.argmin(loss_values)
    loss_min = loss_values[min_loss_index]
    loss_min_values.append(loss_min)

for loss_values, avg_comb in zip(loss_values_list, averages_comb_list):
    min_loss_index = np.argmin(loss_values)
    avg_min = avg_comb[min_loss_index]
    averages_comb_min.append(avg_min)

norm = plt.Normalize(800, 2600)

# Scatter plot
fig = plt.figure()

ax = fig.add_subplot()
ax.grid(False)

scatter = ax.scatter(np.unique(layers_comb_filtered),
                     loss_min_values,
                     c=averages_comb_min,
                     cmap=custom_cmap,
                     norm=norm,
                     alpha=0.95,
                     s=80
                     )


# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of neurons')
cbar.set_ticks(range(800, 2800, 300))


# Connect the dots with a line
ax.plot(np.unique(layers_comb_filtered), loss_min_values, color='black', linestyle='--', alpha=0.20)

ax.tick_params(right=True,
               top=True,
               direction='in',
               length=7)
ax.tick_params(which='minor',
               right=True,
               top=True,
               direction='in',
               length=4)

# Set the number of ticks on the y-axis to 5
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
plt.xticks([2,3,4,5,6,7,8,9,10])

# Labeling axes
ax.set_xlabel('Number of Layers')
ax.set_ylabel('F1-Loss')

# Title
ax.set_title('Relation between F1-Loss and Layers')
plt.show()
plt.savefig("/Users/mariano/Desktop/Work_PhD/PaperNN/Fig.2/mlp_new/MLP_2d_hypertuning_plot_10_layers.pdf")
plt.close()
