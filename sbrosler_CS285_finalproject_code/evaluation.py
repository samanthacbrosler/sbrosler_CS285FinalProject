import importlib
import numpy as np
from numpy import convolve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle
from scipy.cluster.hierarchy import linkage
from scipy.stats import sem
from scipy import signal
from scipy.integrate import trapz
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from RT import savedData, taskData
from gimutil.signal_analysis import erps
from gimutil.configuration import config
from gimutil.visualization import plotting_tools
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from default_model import CnnRnn
from sklearn.metrics import accuracy_score
from preprocessing import *

########################## Evaluating the default model
def evaluate_default(X_test, y_test, default_model, device = torch.device("cuda:0")):
    
    # Convert numpy arrays to torch tensors
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)
    
    with torch.no_grad():
        model = CnnRnn(rnn_dim=253, KS=4, num_layers=2, dropout=0.54, n_targ=30, bidirectional=True, in_channels=253).to(device);
        model.load_state_dict(default_model);
        
        model.eval()
        output = model(X_test)
        _, predicted = torch.max(output, 1)
        
        assert predicted.shape == y_test.shape
        test_accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Default test accuracy: {test_accuracy*100:.2f}%')
        print(f'Chance accuracy: {1/len(np.unique(y_test.cpu().numpy()))*100:.2f}%')
    
    # Compute confusion matrix
    predicted_cm = predicted.cpu().numpy()
    acutal_cm = y_test.cpu().numpy()
    assert predicted_cm.shape == acutal_cm.shape
    default_confmatrix = confusion_matrix(acutal_cm, predicted_cm)
    
    return test_accuracy, default_confmatrix
        

########################## Evaluating the reward model
def evaluate_reward_model(X, y, model, eval_mode = False, device_eval=torch.device("cpu")):
    
    # Convert X to a torch tensor if it's not already a tensor
    if not isinstance(X, torch.Tensor):
        test_X = torch.from_numpy(X).float().to(device_eval)
    else:
        test_X = X.to(device_eval)
    
    # Convert y to a torch tensor if it's not already a tensor
    if not isinstance(y, torch.Tensor):
        test_y = torch.from_numpy(y).long().to(device_eval)
    else:
        test_y = y.to(device_eval)
    
    # Assert that X and y are tensors
    assert torch.is_tensor(test_X), "X is not a tensor"
    assert torch.is_tensor(test_y), "y is not a tensor"
    
    if isinstance(model, dict):
        instance = CnnRnn(rnn_dim=253, KS=4, num_layers=2, dropout=0.54, n_targ=30, bidirectional=True, in_channels=253).to(device_eval)
        instance.load_state_dict(model)
        model = instance
    
    # Initialize lists to store rewards and predictions
    rewards = []
    predictions = []
    
    model.to(device_eval)
    
    if eval_mode == True:
        model.eval()
    
    test_outputs = model(test_X)
    test_qval, _ = torch.max(test_outputs,1)
    
    # Evaluate the model
    with torch.no_grad():
        
        probabilities = torch.softmax(test_outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    
    # Determine the correctness of each prediction
    reward = (predicted == test_y).float().to(device_eval)
        
    for prob, pred, actual in zip(probabilities, predicted, test_y):
        prob_predicted_class = prob[pred].item()  # Probability of the predicted class
        predictions.append(prob_predicted_class)

    # Optionally, print overall test accuracy and chance accuracy
    test_acc = reward.sum().item() / len(test_y)
    
    # Create confusion matrix
    predicted_cm = predicted.cpu().numpy()
    actual_cm = test_y.cpu().numpy()
    assert predicted_cm.shape == actual_cm.shape
    cm = confusion_matrix(actual_cm, predicted_cm)
        
    return reward, test_qval, test_acc, cm

########################## Evaluating the ensemble model
def evaluate_reward_model_ensemble(X, y, models_dict, n_targ = 30, device_eval=torch.device("cuda:2")):
    
    # Convert X to a torch tensor if it's not already a tensor
    if not isinstance(X, torch.Tensor):
        test_X = torch.from_numpy(X).float().to(device_eval)
    else:
        test_X = X.to(device_eval)
    
    # Convert y to a torch tensor if it's not already a tensor
    if not isinstance(y, torch.Tensor):
        test_y = torch.from_numpy(y).long().to(device_eval)
    else:
        test_y = y.to(device_eval)
        
    # Assert that X and y are tensors
    assert torch.is_tensor(test_X), "X is not a tensor"
    assert torch.is_tensor(test_y), "y is not a tensor"
    
    # Initialize lists to store rewards and predictions
    rewards = []
    predictions = []
    
    # Initialize an array to store probabilities from all models
    probabilities = np.zeros((len(models_dict), len(test_X), n_targ))
    
    for idx, (key, model_state_dict) in enumerate(models_dict.items()):
        # Load the model state dict into a new model instance
        model = CnnRnn(rnn_dim=253, KS=4, num_layers=2, dropout=0.54, n_targ=30, bidirectional=True, in_channels=253).to(device_eval)
        model.load_state_dict(model_state_dict)
        model = model.to(device_eval)
        model.eval()

        # Evaluate the model
        with torch.no_grad():
            test_outputs = model(test_X)
            probabilities[idx] = torch.softmax(test_outputs, dim=1).cpu().numpy()
          
    # Average the probabilities across all models
    average_probs = np.mean(probabilities, axis = 0)
    predicted = torch.tensor(np.argmax(average_probs, axis=1), dtype=torch.long, device=device_eval)

    # Determine the correctness of each prediction
    reward = (predicted == test_y).float().to(device_eval)
    for prob, pred, actual in zip(average_probs, predicted, test_y):
        prob_predicted_class = prob[pred].item()  # Probability of the predicted class
        predictions.append(prob_predicted_class)

    # Optionally, print overall test accuracy and chance accuracy
    test_acc = reward.sum().item() / len(test_y)

    # Create confusion matrix
    predicted_cm = predicted.cpu().numpy()
    actual_cm = test_y.cpu().numpy()
    assert predicted_cm.shape == actual_cm.shape
    cm = confusion_matrix(actual_cm, predicted_cm)
        
    return reward, test_acc, cm


######### Hyperparameter tuning for Logistic Regression (Analysis 4)
def n_components_tuning(incorrect_feedback, correct_feedback, n_components_range = np.arange(0.85, 0.99, 0.01), iterations = 3):

    average_accuracies = []
    sem_accuracies = []

    # Iterate over each n_components value
    for n in n_components_range:
        print('n_components value: {:.2f}'.format(n))
        accuracy_history = []

        # Repeat the process for a number of iterations
        for i in range(iterations):

            np.random.seed(i)  # Set the random seed for reproducibility

            # Randomly select indices for creating balanced datasets
            num_samples = incorrect_feedback.shape[0]
            selected_indices_correct = np.random.choice(correct_feedback.shape[0], num_samples, replace=False)
            selected_indices_incorrect = np.random.choice(incorrect_feedback.shape[0], num_samples, replace=False)

            # Create balanced datasets
            balanced_correct_feedback = correct_feedback[selected_indices_correct]
            balanced_incorrect_feedback = incorrect_feedback[selected_indices_incorrect]

            # Combine the datasets and create labels
            combined_feedback = np.concatenate((balanced_incorrect_feedback, balanced_correct_feedback), axis=0)
            labels = np.concatenate((np.zeros(len(balanced_incorrect_feedback)), np.ones(len(balanced_correct_feedback))))

            # Feature extraction
            X_area = area_feature(combined_feedback)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_area, labels, test_size=0.1, stratify=labels)

            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=n, svd_solver='full')
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Train a logistic regression model
            model = LogisticRegression(max_iter=10000)
            model.fit(X_train_pca, y_train)

            # Make predictions and calculate accuracy
            predictions = model.predict(X_test_pca)
            accuracy = accuracy_score(y_test, predictions)
            accuracy_history.append(accuracy)

        # Compute the average accuracy for the current n_components value
        average_accuracy = np.mean(accuracy_history)
        sem_accuracy = np.std(accuracy_history)/np.sqrt(len(accuracy_history))
        sem_accuracies.append(sem_accuracy)
        average_accuracies.append(average_accuracy)
    
    return average_accuracies, sem_accuracies