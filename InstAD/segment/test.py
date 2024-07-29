
import numpy as np
import matplotlib.pyplot as plt

### KNN k
# Data
x_labels = ['k=1', 'k=0.5%', 'k=1%', 'k=5%']
image_level_auc = [64.5, 87.2, 82.0, 69.3]  
pixel_level_auc = [89.9, 98.4, 97.7, 96.2]

# Create a new figure and plot the data
plt.figure()
plt.plot(x_labels, image_level_auc, color='blue', marker='o', label='image-level AUC')
plt.plot(x_labels, pixel_level_auc, color='red', marker='o', label='pixel-level AUC')

# Add title, labels, and legend
plt.title('Selection of k-NN in Zero-Shot Task')
plt.ylabel('AUROC')
plt.ylim(0, 100)  # Set y-axis range from 0 to 100
plt.legend(loc='lower right')  # Set legend position to bottom right
plt.grid(True, linestyle=(0, (5, 5)))  # Show grid

for i, (x_label, y_value) in enumerate(zip(x_labels, image_level_auc)):
    plt.text(i, y_value + 2.5, f'{y_value}', color='blue', ha='center')  # Blue line labels

for i, (x_label, y_value) in enumerate(zip(x_labels, pixel_level_auc)):
    plt.text(i, y_value - 2, f'{y_value}', color='red', ha='center', va='top')

# Show the plot
plt.savefig("/home/anomaly/research/i-patchcore/segment/output/ABLATION_KNN.png")


### Patch Embedding Dimensions
# Data
x_labels = ['dim=128', 'dim=256', 'dim=512', 'dim=1024']
image_level_auc = [79.1, 83.1, 87.2, 83.0]
pixel_level_auc = [97.8, 97.9, 98.4, 98.0]

# Create a new figure and plot the data
plt.figure()
plt.plot(x_labels, image_level_auc, color='blue', marker='o', label='image-level AUC')
plt.plot(x_labels, pixel_level_auc, color='red', marker='o', label='pixel-level AUC')

# Add title, labels, and legend
plt.title('Selection of Patch Embedding Dimensions')
plt.ylabel('AUROC')
plt.ylim(40, 100)  # Set y-axis range from 0 to 100
plt.legend(loc='lower right')  # Set legend position to bottom right
plt.grid(True, linestyle=(0, (5, 5)))  # Show grid

for i, (x_label, y_value) in enumerate(zip(x_labels, image_level_auc)):
    plt.text(i, y_value + 1, f'{y_value}', color='blue', ha='center')  # Blue line labels

for i, (x_label, y_value) in enumerate(zip(x_labels, pixel_level_auc)):
    plt.text(i, y_value - 2, f'{y_value}', color='red', ha='center', va='top')
    
# Show the plot
plt.savefig("/home/anomaly/research/i-patchcore/segment/output/ABLATION_DIM.png")

###
import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['B', 'B + S', 'B + F', 'B + S + F']
image_level_auc = [78.2, 76.5, 81.0, 87.2]
pixel_level_auc = [97.8, 98.1, 97.6, 98.5]

# Combine data for plotting
auc_labels = ['image-level AUC', 'pixel-level AUC']

x = np.arange(len(auc_labels))  # the label locations
width = 0.2  # the width of the bars

# Create a new figure and plot the data
plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()

# Create positions for the bars
positions = np.arange(len(auc_labels))

# Generate colors from a warm colormap
cmap = plt.get_cmap('YlGnBu')  # You can also try 'autumn' or other warm colormaps
colors = [cmap(i / len(methods) + 0.3) for i in range(len(methods))]

# Plotting bars for each method with color gradient
rects1 = ax.bar(positions - 1.5 * width, [image_level_auc[0], pixel_level_auc[0]], width, label=methods[0], color=colors[0])
rects2 = ax.bar(positions - 0.5 * width, [image_level_auc[1], pixel_level_auc[1]], width, label=methods[1], color=colors[1])
rects3 = ax.bar(positions + 0.5 * width, [image_level_auc[2], pixel_level_auc[2]], width, label=methods[2], color=colors[2])
rects4 = ax.bar(positions + 1.5 * width, [image_level_auc[3], pixel_level_auc[3]], width, label=methods[3], color=colors[3])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title('Component-wise Analysis')
ax.set_ylabel('AUROC')
ax.set_xticks(positions)
ax.set_xticklabels(auc_labels)
ax.set_ylim(70, 100)  # Set y-axis range from 60 to 100
ax.legend(loc='lower right')
ax.grid(True, linestyle=(0, (5, 5)))  # Show grid

def autolabel(rects, xpos='center'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x-offset for label alignment

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() * offset[xpos], height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha=ha[xpos], va='bottom')

autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")
autolabel(rects4, "center")

# Save the plot
plt.savefig("/home/anomaly/research/i-patchcore/segment/output/ABLATION_QUMU.png")

###
x_labels = ['B=1', 'B=2', 'B=4', 'B=8', 'B=16']
image_level_auc = [74.2,77.6,80.2,81.2,81.8]  
pixel_level_auc = [97.5,98.1,98.3,98.4,98.5]

# Create a new figure and plot the data
plt.figure(figsize=(8, 6))
plt.plot(x_labels, image_level_auc, color='blue', marker='o', label='image-level AUC')
plt.plot(x_labels, pixel_level_auc, color='red', marker='o', label='pixel-level AUC')

# Add title, labels, and legend
plt.title('Selection of Batch Size in Zero-Shot Task')
plt.ylabel('AUROC')
plt.xlabel('Batch Size')
plt.ylim(60, 100)  # Set y-axis range from 0 to 100
plt.legend(loc='lower right')  # Set legend position to bottom right
plt.grid(True, linestyle=(0, (5, 5)))  # Show grid

for i, (x_label, y_value) in enumerate(zip(x_labels, image_level_auc)):
    plt.text(i, y_value + 2.5, f'{y_value}', color='blue', ha='center')  # Blue line labels

for i, (x_label, y_value) in enumerate(zip(x_labels, pixel_level_auc)):
    plt.text(i, y_value - 2, f'{y_value}', color='red', ha='center', va='top')

# Show the plot
plt.savefig("/home/anomaly/research/i-patchcore/segment/output/ABLATION_BATCH.png")