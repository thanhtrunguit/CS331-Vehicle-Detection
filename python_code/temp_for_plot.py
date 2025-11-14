import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['original', 'copy-n-paste']
max_vehicles = [91, 64]
avg_vehicles = [31.81, 11.18]

# Set up the figure and axis
x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for maximum and average vehicles
rects1 = ax.bar(x - width/2, max_vehicles, width, label='Max Vehicles', color='skyblue')
rects2 = ax.bar(x + width/2, avg_vehicles, width, label='Average Vehicles', color='lightgreen')

# Add text labels above each bar
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}',
                ha='center', va='bottom')

# Customize the chart
ax.set_xlabel('Datasets')
ax.set_ylabel('Number of Vehicles')
ax.set_title('Vehicle Statistics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

# Adjust layout and display
fig.tight_layout()
plt.show()