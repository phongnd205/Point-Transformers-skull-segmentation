import numpy as np
import matplotlib.pyplot as plt


# points_test = np.load('points_test_fold_5.npy')
# labels_target = np.load('labels_target_fold_5.npy')
# labels_predict = np.load('labels_predict_fold_5.npy')



# isubj = 8
# num_point = 2048
# # Example data (replace with your actual data)
# points = points_test[isubj,:,:]  # Example point cloud data
# label = labels_target[isubj*num_point: isubj*num_point +num_point]  # Example labels for each point
# label_predict = labels_predict[isubj*num_point: isubj*num_point +num_point]
# # Define colors for each label
# label_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']

# # Plot target
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot each point with its corresponding color based on the label
# for i in range(11):
#     points_subset = points[:, label == i]
#     color = label_colors[i]
#     ax.scatter(points_subset[0], points_subset[1], points_subset[2], c=color, label=f'Label {i}')

# ax.set_facecolor('none')
# ax.set_axis_off()
# ax.grid(False)
# plt.show()


# # Plot predict
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot each point with its corresponding color based on the label
# for i in range(11):
#     points_subset = points[:, label_predict == i]
#     color = label_colors[i]
#     ax.scatter(points_subset[0], points_subset[1], points_subset[2], c=color, label=f'Label {i}')

# ax.set_facecolor('none')
# ax.set_axis_off()
# ax.grid(False)
# plt.show()


# # Plot error
# # Create label_compare: 1 for correct prediction, 0 for incorrect prediction
# label_compare = (label == label_predict)
# label_compare_np = np.array(label_compare)

# # Define colors for correct and incorrect predictions
# colors = ['g' if val == 1 else 'r' for val in label_compare_np]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the point cloud with colors based on label_compare
# ax.scatter(points[0], points[1], points[2], c=colors)
# ax.set_facecolor('none')
# ax.set_axis_off()
# ax.grid(False)
# plt.show()


loss_train = np.load('log/partseg/Hengshuang/2048_fold5/loss_train.npy')
loss_valid = np.load('log/partseg/Hengshuang/2048_fold5/loss_valid.npy')
acc_train = np.load('log/partseg/Hengshuang/2048_fold5/acc_train.npy')
acc_valid = np.load('log/partseg/Hengshuang/2048_fold5/acc_valid.npy')

x_axis = np.arange(0,len(acc_train))

window_size = 5
acc_valid_padded = np.pad(acc_valid, (window_size//2, window_size//2), mode='edge')
acc_valid_smooth = np.convolve(acc_valid_padded, np.ones(window_size) / window_size, mode='valid')

loss_valid_padded = np.pad(loss_valid, (window_size//2, window_size//2), mode='edge')
loss_valid_smooth = np.convolve(loss_valid_padded, np.ones(window_size) / window_size, mode='valid')


# Create the figure and the first y-axis (accuracy)
fig, ax1 = plt.subplots()
ax1.plot(x_axis, acc_train, color='blue', label='acc_train')
ax1.plot(x_axis, acc_valid_smooth, color = 'red', label='acc_valid')
ax1.set_xlabel('Epochs', fontsize=15)
ax1.set_ylabel('Accuracy', color='blue', fontsize=15)
ax1.tick_params(axis='y', colors='blue')

# Create the second y-axis (loss)
ax2 = ax1.twinx()
ax2.plot(x_axis, loss_train, color='blue', linestyle='--', label='loss_train')
ax2.plot(x_axis, loss_valid_smooth, color='red', linestyle='--', label='loss_valid')
ax2.set_ylabel('Loss', color='red', fontsize=15)
ax2.tick_params(axis='y', colors='red')

# Combine the legend handles and labels from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2)

# Set the title
plt.title('Accuracy and Loss', fontsize=15, fontweight='bold')

# Display the figure
plt.show()