# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# conf_levels = [0.01, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
# e100_map50 = [0.8379, 0.8362, 0.8322, 0.8275, 0.8196, 0.8159, 0.8082]
# e30_map50 = [0.8323, 0.8325, 0.8318, 0.8263, 0.8125, 0.8013, 0.7847]
#
# # Set up the figure and axis
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # Plot lines
# ax.plot(conf_levels, e100_map50, marker='o', label='e100_cygan mAP@0.5', color='blue')
# ax.plot(conf_levels, e30_map50, marker='o', label='e30_cygan mAP@0.5', color='green')
#
# # Add text labels for each point
# for i, (x, y) in enumerate(zip(conf_levels, e100_map50)):
#     ax.text(x, y + 0.002, f'{y:.4f}', ha='center', va='bottom', fontsize=9, color='blue')
# for i, (x, y) in enumerate(zip(conf_levels, e30_map50)):
#     ax.text(x, y - 0.005, f'{y:.4f}', ha='center', va='top', fontsize=9, color='green')
#
# # Set labels and title
# ax.set_xlabel('Confidence Threshold')
# ax.set_ylabel('mAP@0.5')
# ax.set_title('mAP@0.5 Trend Across Confidence Thresholds')
# ax.legend()
#
# # Set x-ticks
# ax.set_xticks(conf_levels)
#
# # Adjust layout and display
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
conf_levels = [0.01, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
e100_f1 = [0.7450, 0.7445, 0.7445, 0.7445, 0.7445, 0.7445, 0.7448]
e30_f1 = [0.7810, 0.7807, 0.7807, 0.7807, 0.7791, 0.7744, 0.7613]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot lines
ax.plot(conf_levels, e100_f1, marker='o', label='e100_cygan F1-score', color='blue')
ax.plot(conf_levels, e30_f1, marker='o', label='e30_cygan F1-score', color='green')

# Add text labels for each point
for i, (x, y) in enumerate(zip(conf_levels, e100_f1)):
    ax.text(x, y + 0.002, f'{y:.4f}', ha='center', va='bottom', fontsize=9, color='blue')
for i, (x, y) in enumerate(zip(conf_levels, e30_f1)):
    ax.text(x, y - 0.005, f'{y:.4f}', ha='center', va='top', fontsize=9, color='green')

# Set labels and title
ax.set_xlabel('Confidence Threshold')
ax.set_ylabel('F1-score')
ax.set_title('F1-score Trend Across Confidence Thresholds')
ax.legend()

# Set x-ticks
ax.set_xticks(conf_levels)

# Adjust layout and display
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()