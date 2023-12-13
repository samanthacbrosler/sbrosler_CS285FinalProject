import os
import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem
import matplotlib.image as mpimg


def calculate_percent_matrices(conf_matrices):
    percent_matrices = np.zeros_like(conf_matrices, dtype=float)
    for k in range(conf_matrices.shape[2]):  # Iterate through each 2D matrix
        for i in range(conf_matrices.shape[0]):  # Iterate through each row
            row_sum = np.sum(conf_matrices[i, :, k])
            if row_sum > 0:  # To avoid division by zero
                percent_matrices[i, :, k] = (conf_matrices[i, :, k] / row_sum) * 100
    return percent_matrices

def plot_avg_performance_bar_chart(default_test_accuracy_history, reward_test_accuracy_history, chance_level = 3.33):
    # Calculate standard error for the mean accuracy of each model
    default_std_error = np.std(default_test_accuracy_history, ddof=1) / np.sqrt(len(default_test_accuracy_history))
    reward_std_error = np.std(reward_test_accuracy_history, ddof=1) / np.sqrt(len(reward_test_accuracy_history))
    
    # Convert accuracies to percentages
    default_mean_accuracy = np.mean(default_test_accuracy_history) * 100
    reward_mean_accuracy = np.mean(reward_test_accuracy_history) * 100
    
    # Plot the bar chart
    plt.figure(figsize=(7, 6))
    plt.bar('Default', default_mean_accuracy, yerr=default_std_error * 100, capsize=5, color='lightgray', alpha=0.7)
    plt.bar('Reward', reward_mean_accuracy, yerr=reward_std_error * 100, capsize=5, color='steelblue', alpha=0.7)

    # Add chance level line
    plt.axhline(y=chance_level, color='black', linestyle='--', label='Chance')

    # Set the font sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Test Accuracy (%)', fontsize=18)
    plt.title('Model Performance Comparison', fontsize=18)

    # Add legend
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()

def plot_class_wise_bar_chart(default_confmatrix_all, reward_confmatrix_all, class_labels):
    
    # Calculate percent matrices and SEM for the default model
    percent_matrices = calculate_percent_matrices(default_confmatrix_all)
    default_sem = np.diag(stats.sem(percent_matrices, axis=2))
    default_class_accs = np.diag(percent_matrices.mean(axis=2))
    
    # Calculate percent matrices and SEM for the reward model
    percent_matrices = calculate_percent_matrices(reward_confmatrix_all)
    reward_sem = np.diag(stats.sem(percent_matrices, axis=2))
    reward_class_accs = np.diag(percent_matrices.mean(axis=2))
    
    n_classes = len(default_class_accs)
    
    # Width of the bars
    bar_width = 0.35
    
    # Set positions of the bars
    default_bar_positions = np.arange(n_classes)
    reward_bar_positions = [x + bar_width for x in default_bar_positions]
    
    # Create bar chart
    plt.figure(figsize=(18, 6))
    plt.bar(default_bar_positions, default_class_accs, width=bar_width, color='lightgray', label='Default Model', yerr=default_sem, capsize=0)
    plt.bar(reward_bar_positions, reward_class_accs, width=bar_width, color='steelblue', label='Reward Model', yerr=reward_sem, capsize=0)
    
    # Add labels, title, etc.
    plt.xlabel('Class', fontsize=22)
    plt.ylabel('Average Accuracy (%)', fontsize=22)
    plt.title('Class-wise Accuracy Comparison', fontsize=22)
    plt.xticks([r + bar_width / 2 for r in range(n_classes)], class_labels, fontsize=18, rotation=90)
    plt.yticks(fontsize=18)
    
    plt.legend(fontsize=14)
     
    plt.show()

def plot_average_clustered_conf_matrix(confmatrix_all, test_acc_mean, val_acc_mean, class_names):
    
    # Convert elements in confusion matrices into percentages
    percent_matrices = calculate_percent_matrices(confmatrix_all)
    
    # Calculate average across test folds
    cm_mean = np.mean(percent_matrices, axis=2)
    
    # Plot clustered confusion matrix
    cluster_grid = sns.clustermap(cm_mean, cmap='viridis', linewidths=0, metric="euclidean", method='average', square=True, annot=False, xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 14},
                                  vmin=0, vmax=90, cbar_kws={'orientation': 'horizontal', 'pad': 2})
    
    # Adjust heatmap layout
    cluster_grid.fig.subplots_adjust(bottom=0.15, top=1.2)
    ax = cluster_grid.ax_heatmap
    ax.tick_params(axis='x', labelrotation=90, labelsize=18)
    ax.tick_params(axis='y', labelrotation=0, labelsize=18)
    ax.set_xlabel("Predicted", fontsize=20, labelpad=10)
    ax.set_ylabel("Actual", fontsize=20, labelpad=10)
    plt.subplots_adjust(bottom=0.55, top=1.3)
    
    # Populate title text
    accuracy_text = 'Test Acc: {:.3f}%,\nVal Acc: {:.3f}%, Chance: {:.3f}%'.format(
        test_acc_mean * 100, val_acc_mean * 100, (1 / len(class_names)) * 100)
    ax.set_title('\n {}'.format(accuracy_text), size=22, pad=130)
    
    # Adjust colorbar
    cbar = cluster_grid.cax
    cbar.set_position([0.1, 0.01, 0.8, 0.04])
    cbar.tick_params(labelsize=20)
    

def plot_triple_comparison_bar_chart(default_test_accuracy_history, reward_test_accuracy_history, third_test_accuracy_history, third_label = 'enter label', third_color = 'orange', chance = 3.33):
    default_sem = np.std(default_test_accuracy_history, ddof=1) / np.sqrt(len(default_test_accuracy_history))
    reward_sem = np.std(reward_test_accuracy_history, ddof=1) / np.sqrt(len(reward_test_accuracy_history))
    third_sem = np.std(third_test_accuracy_history, ddof=1) / np.sqrt(len(third_test_accuracy_history))

    plt.figure(figsize = (7,6))
    plt.bar('Default', np.mean(default_test_accuracy_history)*100, yerr=default_sem*100, capsize=5, color='lightgray', alpha=0.7)
    plt.bar('Reward', np.mean(reward_test_accuracy_history)*100, yerr=reward_sem*100, capsize=5, color='steelblue', alpha=0.7)
    plt.bar(third_label, np.mean(third_test_accuracy_history)*100, yerr=third_sem*100, capsize=5, color=third_color, alpha=0.7)

    plt.axhline(y = chance, color='black', linestyle='--', label = 'Chance')

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.ylabel('Test Accuracy (%)', fontsize = 18)
    plt.title('Model Performance Comparison', fontsize = 18)

    plt.legend(fontsize = 12)
    

def calculate_average_metric(all_test_splits_metrics, model_type, metric, num_points=30): # Used for ablation experiment (figure 2)

    # Normalize each run's epochs to a percentage scale (0-100%)
    normalized_metric_values = []
    count = 0
    for split in all_test_splits_metrics.values():
        count+=1
        best_val_fold = 0
        max_val_acc_folds = 0.0

        for fold in split[model_type]:
            
            # Identify the highest performing fold based on validation accuracy
            max_val_acc = np.max(split[model_type][fold]['Validation Accuracy'])
            
            if max_val_acc > max_val_acc_folds:
                max_val_acc_folds = max_val_acc
                best_val_fold = fold
        
        epochs = np.linspace(0, 1, num=len(split[model_type][best_val_fold][metric]))
        metric_values = split[model_type][best_val_fold][metric]
        
        # Interpolate to a common scale
        f = interpolate.interp1d(epochs, metric_values, kind='linear')
        common_epochs = np.linspace(0, 1, num=num_points)
        normalized_metric_values.append(f(common_epochs))
        
    # Convert the list of normalized metrics to a 2D numpy array
    normalized_metric_array = np.array(normalized_metric_values, dtype=float)
    
    # Calculate the average and SEM across the normalized values
    average_values = np.mean(normalized_metric_array, axis=0)
    sem_values = np.std(normalized_metric_array, axis=0) / np.sqrt(len(normalized_metric_values))
    
    if metric == 'Validation Accuracy' or metric == 'Training Accuracy':
        average_values = average_values*100
        sem_values = sem_values*100
        
    return average_values, sem_values, common_epochs

def plot_ablation_training_validation_accuracy(normalized_epochs, metrics_data, titles, colors):

    # Plotting setup
    fig, axes = plt.subplots(1, len(metrics_data), figsize=(18, 4), sharey=True)

    # Loop through the provided metrics data to create each subplot
    for i, ((train_means, train_sems, val_means, val_sems), title, color) in enumerate(zip(metrics_data, titles, colors)):
        axes[i].plot(normalized_epochs, train_means, color=color, linestyle='--', linewidth=2, label='Training')
        axes[i].fill_between(normalized_epochs, train_means - train_sems, train_means + train_sems, color=color, alpha=0.2)
        axes[i].plot(normalized_epochs, val_means, color=color, marker='o', markersize=8, linewidth=0, label='Validation')
        axes[i].fill_between(normalized_epochs, val_means - val_sems, val_means + val_sems, color=color, alpha=0.2)
        axes[i].set_title(title, fontsize=22)
        axes[i].set_xlabel('Normalized Epochs', fontsize=18)
        axes[i].legend(loc='upper left', fontsize=14)
        axes[i].tick_params(labelsize=18)

    # Set common y-axis labels
    axes[0].set_ylabel('Average Accuracy (%)', fontsize=18)

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def plot_sorted_confusion_matrix(conf_matrices, test_acc_mean, val_acc_mean, class_labels):
    percent_matrices = calculate_percent_matrices(conf_matrices)

    # Take mean across 10 matrices
    cm_mean = np.mean(percent_matrices, axis = 2)

    # Now sort
    diag = np.diag(cm_mean)

    # Sort the diagonal elements in descending order
    sorted_indices = np.argsort(diag)[::-1]

    # Rearrange the rows and columns of the confusion matrix
    sorted_cm_mean = cm_mean[sorted_indices][:, sorted_indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    label_fontsize = 18
    tick_fontsize = 18
    title_fontsize = 26

    conf_mat = sns.heatmap(sorted_cm_mean, cmap='viridis', linewidth=0, annot=False, fmt=".1%", cbar=True, vmin = 0, vmax = 90)
    conf_mat.set_aspect('equal')
    model_classes = [class_labels[i] for i in sorted_indices]

    # Set axis labels
    ax.set_ylabel('Actual label', fontsize=label_fontsize+8)
    ax.set_xlabel('Predicted label', fontsize=label_fontsize+8)

    # Configure axis tick labels and positions
    ax.set_xticks(np.arange(len(model_classes)) + 0.5)
    ax.set_xticklabels(model_classes, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(np.arange(len(model_classes)) + 0.5)
    ax.set_yticklabels(model_classes, rotation=0, fontsize=tick_fontsize)
    ax.xaxis.set_tick_params(labeltop=False, top=False)
    ax.yaxis.set_tick_params(labelright=False, right=False)

    # Populate title text
    if val_acc_mean == None:
        accuracy_text = 'Test Acc: {:.3f}%,\nChance: {:.3f}%'.format(
        test_acc_mean * 100, (1 / len(model_classes)) * 100)
    else:
        accuracy_text = 'Test Acc: {:.3f}%,\nVal Acc: {:.3f}%, Chance: {:.3f}%'.format(
        test_acc_mean * 100, val_acc_mean * 100, (1 / len(model_classes)) * 100)
    ax.set_title('\n {}'.format(accuracy_text), size=title_fontsize)

    # Set color bar label to show percentages
    cbar = conf_mat.collections[0].colorbar
    cbar.set_label('Percentage (%)', fontsize=24)
    cbar.ax.tick_params(labelsize=tick_fontsize)

def plot_acc_vs_reward_error(default_test_accuracy_history, reward_test_accuracy_dict, ensemble_test_accuracy_dict):
    
    reward_error = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    
    mean_reward = []
    sterr_reward = []
    
    mean_ensemble = []
    sterr_ensemble = []

    for index, p in enumerate(reward_error):

        vals_reward = reward_test_accuracy_dict[p]
        mean_reward.append(np.mean(vals_reward))
        sterr_reward.append(np.std(vals_reward, ddof=1) / np.sqrt(len(vals_reward)))
        
        vals_ensemble = ensemble_test_accuracy_dict[p]
        mean_ensemble.append(np.mean(vals_ensemble))
        sterr_ensemble.append(np.std(vals_ensemble, ddof=1) / np.sqrt(len(vals_ensemble)))

    mean_ensemble = np.array(mean_ensemble)
    sterr_ensemble = np.array(sterr_ensemble)
    
    mean_reward = np.array(mean_reward)
    sterr_reward = np.array(sterr_reward)

    default_mean = np.mean(default_test_accuracy_history)
    default_mean = np.ones(len(reward_error)) * default_mean
    default_sterr = np.std(default_test_accuracy_history, ddof=1) / np.sqrt(len(default_test_accuracy_history))

    plt.figure(figsize = (8,5))
    plt.plot(reward_error, mean_reward*100, color = 'steelblue', marker = '.', markersize = 16, label = 'Reward model')
    plt.fill_between(reward_error, (mean_reward - sterr_reward)*100, (mean_reward + sterr_reward)*100, color = 'steelblue', alpha=0.2)

    plt.plot(reward_error, mean_ensemble*100, color = 'darkseagreen', marker = '.', markersize = 16, label = 'Ensemble model')
    plt.fill_between(reward_error, (mean_ensemble - sterr_ensemble)*100, (mean_ensemble + sterr_ensemble)*100, color = 'darkseagreen', alpha=0.2)

    plt.plot(reward_error, default_mean*100, color = 'gray', markersize = 16, label = 'Default model')
    plt.fill_between(reward_error, (default_mean - default_sterr)*100, (default_mean + default_sterr)*100, color = 'gray', alpha=0.2)
    plt.title('Sensitivity to Mislabeled Rewards', fontsize = 22)

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.legend(frameon=False, fontsize = 16)
    plt.ylabel('Test Accuracy (%)', fontsize = 18)
    plt.xlabel('Reward Mislabeling Rate', fontsize = 18)
    
def plot_erp_b3_gaussiansmooth(erp_1, sem_1, label1, erp_2, sem_2, label2, window, start_time, end_time):
    
    # Inputs erp1 and erp2 are in the shape of trials x time x channels
    
    with open('data/b3_layout.pkl','rb') as f:
        elec_layout = pickle.load(f)
   
    # Data
    t = np.linspace(start_time, end_time, erp_1.shape[0]) 

#     # ERPs and SEMs
#     erp_1 = np.mean(erp1, axis=0)
#     sem_1 = sem(erp1, axis = 0)

#     erp_2 = np.mean(erp2, axis=0)
#     sem_2 = sem(erp2, axis = 0)

    # Apply moving average smoothing to the data
    erp_1_smooth = np.zeros_like(erp_1)
    erp_2_smooth = np.zeros_like(erp_2)
    sem_1_smooth = np.zeros_like(sem_1)
    sem_2_smooth = np.zeros_like(sem_2)

    for i in range(erp_1.shape[1]):
        erp_1_smooth[:,i] = gaussian_filter1d(erp_1[:,i], window)
        erp_2_smooth[:,i] = gaussian_filter1d(erp_2[:,i], window)
        sem_1_smooth[:,i] = gaussian_filter1d(sem_1[:,i], window)
        sem_2_smooth[:,i] = gaussian_filter1d(sem_2[:,i], window)

    fig, axes = plt.subplots(elec_layout.shape[0], elec_layout.shape[1], figsize=(30, 50), sharey=True)

    # Adjust the margins and plot spacing
    fig.subplots_adjust(wspace=0.000, hspace=0.000,
                        left=0.04, right=0.99,
                        top=0.95, bottom=0.02)

    for r in range(0, elec_layout.shape[0]):
        for c in range(0, elec_layout.shape[1]):
            axes[r, c].axhline(y=0, color='k', alpha=0.5, linewidth=0.5)
            axes[r, c].axvline(x=0, color='k', alpha=0.5, linewidth=0.5)
            axes[r,c].plot(t, erp_1_smooth[:, elec_layout[r,c]], color='orange',linewidth = 3, label = label1)
            axes[r,c].plot(t, erp_2_smooth[:, elec_layout[r,c]], color='cornflowerblue',linewidth = 3, label = label2)
            axes[r,c].fill_between(t, erp_1_smooth[:, elec_layout[r,c]] + sem_1_smooth[:, elec_layout[r,c]], erp_1_smooth[:, elec_layout[r,c]] - sem_1_smooth[:, elec_layout[r,c]], color='orange', alpha=0.2)
            axes[r,c].fill_between(t, erp_2_smooth[:, elec_layout[r,c]] + sem_2_smooth[:, elec_layout[r,c]], erp_2_smooth[:, elec_layout[r,c]] - sem_2_smooth[:, elec_layout[r,c]], color='cornflowerblue', alpha=0.2) 
            #axes[r,c].set_ylim([-0.5,1.0])
            axes[r, c].set_title(f' e{elec_layout[r,c]}', y=0.78, fontsize=20, loc='left')
            
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.00, 0.95), fontsize=50)
            
def calc_and_plot_erp(erp1, label1, erp2, label2, window, start_time, end_time):
    
    # Inputs erp1 and erp2 are in the shape of trials x time x channels
    
    with open('data/b3_layout.pkl','rb') as f:
        elec_layout = pickle.load(f)
   
    # Data
    t = np.linspace(start_time, end_time, erp1.shape[1]) 

    # ERPs and SEMs
    erp_1 = np.mean(erp1, axis=0)
    sem_1 = sem(erp1, axis = 0)

    erp_2 = np.mean(erp2, axis=0)
    sem_2 = sem(erp2, axis = 0)

    # Apply moving average smoothing to the data
    erp_1_smooth = np.zeros_like(erp_1)
    erp_2_smooth = np.zeros_like(erp_2)
    sem_1_smooth = np.zeros_like(sem_1)
    sem_2_smooth = np.zeros_like(sem_2)

    for i in range(erp_1.shape[1]):
        erp_1_smooth[:,i] = gaussian_filter1d(erp_1[:,i], window)
        erp_2_smooth[:,i] = gaussian_filter1d(erp_2[:,i], window)
        sem_1_smooth[:,i] = gaussian_filter1d(sem_1[:,i], window)
        sem_2_smooth[:,i] = gaussian_filter1d(sem_2[:,i], window)

    fig, axes = plt.subplots(elec_layout.shape[0], elec_layout.shape[1], figsize=(30, 50), sharey=True)

    # Adjust the margins and plot spacing
    fig.subplots_adjust(wspace=0.000, hspace=0.000,
                        left=0.04, right=0.99,
                        top=0.95, bottom=0.02)

    for r in range(0, elec_layout.shape[0]):
        for c in range(0, elec_layout.shape[1]):
            axes[r, c].axhline(y=0, color='k', alpha=0.5, linewidth=0.5)
            axes[r, c].axvline(x=0, color='k', alpha=0.5, linewidth=0.5)
            axes[r,c].plot(t, erp_1_smooth[:, elec_layout[r,c]], color='orange',linewidth = 3, label = label1)
            axes[r,c].plot(t, erp_2_smooth[:, elec_layout[r,c]], color='cornflowerblue',linewidth = 3, label = label2)
            axes[r,c].fill_between(t, erp_1_smooth[:, elec_layout[r,c]] + sem_1_smooth[:, elec_layout[r,c]], erp_1_smooth[:, elec_layout[r,c]] - sem_1_smooth[:, elec_layout[r,c]], color='orange', alpha=0.2)
            axes[r,c].fill_between(t, erp_2_smooth[:, elec_layout[r,c]] + sem_2_smooth[:, elec_layout[r,c]], erp_2_smooth[:, elec_layout[r,c]] - sem_2_smooth[:, elec_layout[r,c]], color='cornflowerblue', alpha=0.2) 
            #axes[r,c].set_ylim([-0.5,1.0])
            axes[r, c].set_title(f' e{elec_layout[r,c]}', y=0.78, fontsize=20, loc='left')
            
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.00, 0.95), fontsize=50)
    
def plot_n_components_tuning(average_accuracies, sem_accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(0.85, 0.99, 0.01), np.array(average_accuracies)*100, color = 'steelblue', marker='o')
    plt.fill_between(np.arange(0.85, 0.99, 0.01), np.array(average_accuracies)*100 - np.array(sem_accuracies)*100, np.array(average_accuracies)*100 + np.array(sem_accuracies)*100, color='steelblue', alpha=0.2)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('n_components in PCA', fontsize = 18)
    plt.ylabel('Average Accuracy (%)', fontsize = 18)
    plt.title('Average Accuracy vs. Number of PCA Components ', fontsize = 22)
    plt.show()
    

def plot_relative_electrode_strength(erp_incorrect_feedback, erp_correct_feedback):
    # This function will throw an error since you don't have access to a special plotting library I share with my lab.
    # If the code under "try:" doesn't work, it will load in the saved .png

    try: 
        erp = np.max(erp_incorrect_feedback - erp_correct_feedback, axis=0)

        # Parameters
        subject      = 'bravo3'
        fig_params   = dict(figsize=(10., 10.), dpi=100.)
        color_params = {'min': 0., 'max': 1., 'relative': True}
        color_spec   = 'Blues'
        weights      = erp
        cbar_params  = {
            'plot_colorbar'  : True,
            'colorbar_title' : 'Relative Electrode Activity',
        }

        # Creates the plot
        all_plot_params = config.load_image_and_elec_config(subject)
        all_plot_params['figure_params'] = fig_params
        all_plot_params['elec_size_color_params']['color_spec'] = color_spec
        all_plot_params['elec_size_color_params']['color_params'] = color_params
        all_plot_params['elec_weights'] = weights
        all_plot_params.update(cbar_params)
        plotting_tools.plot_images_and_elecs(**all_plot_params)

    except Exception as e:

        #print(f"An error occurred: {e}")
        img = mpimg.imread('data/erp_brain_heatmap.png')
        plt.imshow(img)
        plt.axis('off')  # Turn off axis numbers and labels
        plt.show()
