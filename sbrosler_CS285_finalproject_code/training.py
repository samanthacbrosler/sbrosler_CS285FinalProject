import copy
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
from preprocessing import *
from evaluation import *
from default_model import CnnRnn
from sklearn.metrics import accuracy_score

######################### Train default
def train_default(X, 
                  y_num, 
                  device = torch.device("cuda:0"),
                  batch_size = 60, 
                  num_epochs = 30, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 5,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    fold_metrics = {fold: {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []} for fold in range(val_numfolds)}
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_loss = float('inf')
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        # Initialize default CNN-RNN model and move to device
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_loss = float('inf')  # Track the best validation loss
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            # Training phase
            model.train()
            for i in range(0, len(fold_train_X), batch_size):
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted_labels = torch.max(outputs, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / (len(fold_train_X) / batch_size)
            average_accuracy = correct_predictions / total_samples
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(fold_valid_X)
                
                # Compute validation loss for early stopping
                val_loss = criterion(val_outputs, fold_valid_y)
                
                # Compute validation accuracy
                _, val_predicted_labels = torch.max(val_outputs, dim=1)
                val_correct_predictions = (val_predicted_labels == fold_valid_y).sum().item()
                val_accuracy = val_correct_predictions / fold_valid_y.size(0)
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_accuracy)
                    
            if val_loss < best_val_loss:
                assert val_loss == val_loss_history[-1]
                best_val_loss = val_loss
                best_model_val_acc = val_accuracy_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                
                patience_counter = 0
                
                # Keep track of lowest loss across folds for selecting best default model across validation folds
                if val_loss < overall_best_val_loss:
                    overall_best_val_loss = val_loss
                    overall_best_model_val_acc = best_model_val_acc
                    best_model = copy.deepcopy(model.state_dict()) 
                    
            else:
                patience_counter +=1
                if patience_counter >= patience:
                    print("     Early stopping! No improvement in validation loss for", patience, "epochs.")
                    break
            
        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_val_loss:.4f}, Val Accuracy: {best_model_val_acc:.4f}")
        
        # This code stores training loss and validation accuracy up to the point of early stopping for each validation fold. The metrics are recorded and stored for each epoch up until the early stopping point. If the best model for a fold is determined before the patience limit is reached, epochs beyond this point (where the model might have started overfitting) will not have their metrics recorded.
        fold_metrics[fold]['Training Loss'] = train_loss_history
        fold_metrics[fold]['Training Accuracy'] = train_accuracy_history
        fold_metrics[fold]['Validation Accuracy'] = val_accuracy_history
        fold_metrics[fold]['Validation Loss'] = val_loss_history
        
    return best_model, overall_best_model_val_acc, fold_metrics


######################### Train reward
def train_reward(X, 
                  y_num, 
                  default_model,
                  device = torch.device("cpu"),
                  batch_size = 60, 
                  num_epochs = 30, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 10,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    fold_metrics = {fold: {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []} for fold in range(val_numfolds)}
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_accuracy = 0.0
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        # Initialize CNN-RNN model, move to device, and pretrain with default model weights
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        model.load_state_dict(default_model)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_acc = 0.0  # Track the best validation accuracy
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
            #print('epoch:', epoch)
            total_loss = 0.0
            correct_predictions = 0
            num_batches = 0
            
            model.train()
            # Training phase
            for i in range(0, len(fold_train_X), batch_size):
                num_batches += 1
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                # Get reward and predictions
                reward, predictions, test_acc, _ = evaluate_reward_model(inputs, labels, model, device_eval=torch.device(device))
                
                optimizer.zero_grad()
                
                assert predictions.device == reward.device, "Predictions and reward are on different devices"
                loss = criterion(predictions.to(device), reward.to(device))
                
                # Making sure the model is on the device and in training mode 
                model.train()
                assert next(model.parameters()).device == device, "Model is not on the expected device"
                assert model.training, "Model is not in training mode"
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_predictions += test_acc
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / num_batches
            average_accuracy = correct_predictions / num_batches
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            with torch.no_grad():
                val_reward, val_predictions, val_acc, _ = evaluate_reward_model(fold_valid_X, fold_valid_y, model, eval_mode = True, device_eval=torch.device(device))
                val_loss = criterion(val_predictions.to(device), val_reward.to(device))
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_acc)
                
            # Early stopping if val acc doesn't improve for 5 epochs    
            if val_acc > best_val_acc:
                assert val_acc == val_accuracy_history[-1]
                best_val_acc = val_accuracy_history[-1]
                best_model_val_loss = val_loss_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                
                patience_counter = 0
                
                if val_acc > overall_best_val_accuracy:
                    overall_best_val_accuracy = val_acc # best val acc across validation folds for a single test fold
                    best_model = copy.deepcopy(model.state_dict()) 
                    
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping! No improvement in validation accuracy for", patience, "epochs.")
                    break
            
                    
        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_model_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
        
        fold_metrics[fold]['Training Loss'] = train_loss_history
        fold_metrics[fold]['Training Accuracy'] = train_accuracy_history
        fold_metrics[fold]['Validation Accuracy'] = val_accuracy_history
        fold_metrics[fold]['Validation Loss'] = val_loss_history
        
    return best_model, overall_best_val_accuracy, fold_metrics

######################### Train reward no prior
def train_reward_no_prior(X, 
                  y_num, 
                  device = torch.device("cpu"),
                  batch_size = 60, 
                  num_epochs = 50, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 10,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    fold_metrics = {fold: {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []} for fold in range(val_numfolds)}
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_accuracy = 0.0
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        # Initialize CNN-RNN model, move to device
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        #test_uniform_distribution(model, device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_acc = 0.0  # Track the best validation accuracy
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
            #print('epoch:', epoch)
            total_loss = 0.0
            correct_predictions = 0
            num_batches = 0
            
            model.train()
            # Training phase
            for i in range(0, len(fold_train_X), batch_size):
                num_batches += 1
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                # Get reward and predictions
                reward, predictions, test_acc, _ = evaluate_reward_model(inputs, labels, model, device_eval=torch.device(device))
                
                optimizer.zero_grad()
                
                assert predictions.device == reward.device, "Predictions and reward are on different devices"
                loss = criterion(predictions.to(device), reward.to(device))
                
                # move model back to device and set model back in training mode after evaluate_reward_model function
                model.train()
                assert next(model.parameters()).device == device, "Model is not on the expected device"
                assert model.training, "Model is not in training mode"
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_predictions += test_acc
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / num_batches
            average_accuracy = correct_predictions / num_batches
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            with torch.no_grad():
                val_reward, val_predictions, val_acc, _ = evaluate_reward_model(fold_valid_X, fold_valid_y, model, eval_mode = True, device_eval=torch.device(device))
                val_loss = criterion(val_predictions.to(device), val_reward.to(device))
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_acc)
                
            # Early stopping if val acc doesn't improve for 5 epochs    
            if val_acc > best_val_acc:
                assert val_acc == val_accuracy_history[-1]
                best_val_acc = val_accuracy_history[-1]
                best_model_val_loss = val_loss_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                
                patience_counter = 0
                #print('Updated best validation accuracy for this current validation fold:')
                
                if val_acc > overall_best_val_accuracy:
                    overall_best_val_accuracy = val_acc # best val acc across validation folds for a single test fold
                    best_model = copy.deepcopy(model.state_dict()) 
                    #print(best_model['preprocessing_conv.weight'][:10,0,0])
                    #print('****Updated best model across validation folds for current test fold!!!! :)')
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping! No improvement in validation accuracy for", patience, "epochs.")
                    break
            
                    
        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_model_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
        
        fold_metrics[fold]['Training Loss'] = train_loss_history
        fold_metrics[fold]['Training Accuracy'] = train_accuracy_history
        fold_metrics[fold]['Validation Accuracy'] = val_accuracy_history
        fold_metrics[fold]['Validation Loss'] = val_loss_history
        
    return best_model, overall_best_val_accuracy, fold_metrics

######################### Train reward with mislabeled reward signal
def train_reward_error(X, 
                  y_num, 
                  default_model,
                  p,
                  repeats,
                  device = torch.device("cpu"),
                  batch_size = 60, 
                  num_epochs = 30, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 10,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_accuracy = 0.0
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        # Initialize CNN-RNN model, move to device, and pretrain with default model weights
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        model.load_state_dict(default_model)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_acc = 0.0  # Track the best validation accuracy
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
        
            total_loss = 0.0
            correct_predictions = 0
            num_batches = 0
            
            model.train()
            # Training phase
            for i in range(0, len(fold_train_X), batch_size):
                
                num_batches += 1
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                # Get reward and predictions
                reward, predictions, test_acc, _ = evaluate_reward_model(inputs, labels, model, device_eval=torch.device(device))
                
                # Randomly make batch_size * p trials incorrect
                random.seed((repeats+1)*(i+1)) # sets a different random seed for each repeat and batch but keeps seed consistent across epochs
                num_incorrect_trials = int(batch_size * p)
                selected_indices = random.sample(range(len(reward)), num_incorrect_trials)
                reward[selected_indices] = 1-reward[selected_indices]
                
                optimizer.zero_grad()
                
                assert predictions.device == reward.device, "Predictions and reward are on different devices"
                
                loss = criterion(predictions.to(device), reward.to(device))
                
                # make sure the model is on the device and in training mode
                model.train()
                assert next(model.parameters()).device == device, "Model is not on the expected device"
                assert model.training, "Model is not in training mode"
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_predictions += test_acc
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / num_batches
            average_accuracy = correct_predictions / num_batches
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            with torch.no_grad():
                val_reward, val_predictions, val_acc, _ = evaluate_reward_model(fold_valid_X, fold_valid_y, model, eval_mode = True, device_eval=torch.device(device))
                val_loss = criterion(val_predictions.to(device), val_reward.to(device))
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_acc)
                
            
            # Early stopping if val acc doesn't improve for 5 epochs    
            if val_acc > best_val_acc:
                assert val_acc == val_accuracy_history[-1]
                best_val_acc = val_accuracy_history[-1]
                best_model_val_loss = val_loss_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                
                patience_counter = 0
                
                if val_acc > overall_best_val_accuracy:
                    overall_best_val_accuracy = val_acc # best val acc across validation folds for a single test fold
                    best_model = copy.deepcopy(model.state_dict()) 
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping! No improvement in validation accuracy for", patience, "epochs.")
                    break

                    
        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_model_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
        
    return best_model, overall_best_val_accuracy
                
                

######################### Train ensemble reward
def train_reward_ensemble(X, 
                  y_num, 
                  default_model,
                  device = torch.device("cpu"),
                  batch_size = 60, 
                  num_epochs = 30, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 10,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    fold_metrics = {fold: {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []} for fold in range(val_numfolds)}
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_accuracy = 0.0
    
    # Store best model for each fold 
    best_fold_model = {}
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        
        # Initialize CNN-RNN model, move to device, and pretrain with default model weights
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        model.load_state_dict(default_model)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_acc = 0.0  # Track the best validation accuracy
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
            #print('epoch:', epoch)
            total_loss = 0.0
            correct_predictions = 0
            num_batches = 0
            
            model.train()
            # Training phase
            for i in range(0, len(fold_train_X), batch_size):
                num_batches += 1
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                # Get reward and predictions
                reward, predictions, test_acc, _ = evaluate_reward_model(inputs, labels, model, device_eval=torch.device(device))
                
                optimizer.zero_grad()
                
                assert predictions.device == reward.device, "Predictions and reward are on different devices"
                loss = criterion(predictions.to(device), reward.to(device))
                
                # make sure model is in training mode and on the device
                model.train()
                assert next(model.parameters()).device == device, "Model is not on the expected device"
                assert model.training, "Model is not in training mode"
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_predictions += test_acc
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / num_batches
            average_accuracy = correct_predictions / num_batches
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            with torch.no_grad():
                val_reward, val_predictions, val_acc, _ = evaluate_reward_model(fold_valid_X, fold_valid_y, model, eval_mode = True, device_eval=torch.device(device))
                val_loss = criterion(val_predictions.to(device), val_reward.to(device))
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_acc)
            
            # Early stopping if val loss doesn't improve for 5 epochs    
            if val_acc > best_val_acc:
                assert val_acc == val_accuracy_history[-1]
                best_val_acc = val_accuracy_history[-1]
                best_model_val_loss = val_loss_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                best_fold_model[fold] = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping! No improvement in validation accuracy for", patience, "epochs.")
                    break

        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_model_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
        
        fold_metrics[fold]['Training Loss'] = train_loss_history
        fold_metrics[fold]['Training Accuracy'] = train_accuracy_history
        fold_metrics[fold]['Validation Accuracy'] = val_accuracy_history
        fold_metrics[fold]['Validation Loss'] = val_loss_history
        
    return best_fold_model, fold_metrics

######################### Train ensemble with mislabeled reward signal
def train_reward_error_ensemble(X, 
                  y_num, 
                  default_model,
                  p,
                  repeats,
                  device = torch.device("cpu"),
                  batch_size = 60, 
                  num_epochs = 30, 
                  rnn_dim = 253,
                  KS = 4,
                  num_layers = 2,
                  dropout = 0.54,
                  n_targ = 30,
                  bidirectional = True, 
                  in_channels = 253,
                  patience = 10,
                  val_numfolds = 10):
    
    print(f"Using device: {device}")
    
    fold_metrics = {fold: {'Training Loss': [], 'Training Accuracy': [], 'Validation Loss': [], 'Validation Accuracy': []} for fold in range(val_numfolds)}
    
    # Cross-validation setup
    val_skf = StratifiedKFold(n_splits = val_numfolds, shuffle = True, random_state=42)
    
    # Convert numpy arrays to torch tensors
    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y_num).long()
    
    # Storing metrics across validation splits
    best_model = None
    overall_best_val_accuracy = 0.0
    
    # Store best model for each fold 
    best_fold_model = {}
    
    # Validation folds
    for fold, (train_index, val_index) in enumerate(val_skf.split(X_torch, y_torch)):
        print(f"     \nValidation Fold {fold + 1}/{val_numfolds}:")
        
        # Split data into train and validation sets and move to device
        fold_train_X, fold_valid_X = X_torch[train_index].to(device), X_torch[val_index].to(device)
        fold_train_y, fold_valid_y = y_torch[train_index].to(device), y_torch[val_index].to(device)
        
        
        # Initialize CNN-RNN model, move to device, and pretrain with default model weights
        model = CnnRnn(rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels).to(device)
        model.load_state_dict(default_model)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create empty lists to store metrics for this fold
        train_loss_history, train_accuracy_history = [], []
        val_loss_history, val_accuracy_history = [], []
        
        # Track metrics for early stopping
        best_val_acc = 0.0  # Track the best validation accuracy
        patience_counter = 0  # Track the number of epochs without improvement
        
        # Training loop for this fold
        for epoch in range(num_epochs):
            #print('epoch:', epoch)
            total_loss = 0.0
            correct_predictions = 0
            num_batches = 0
            
            model.train()
            # Training phase
            for i in range(0, len(fold_train_X), batch_size):
                num_batches += 1
                inputs = fold_train_X[i:i+batch_size]
                labels = fold_train_y[i:i+batch_size]
                
                # Get reward and predictions 
                reward, predictions, test_acc, _ = evaluate_reward_model(inputs, labels, model, device_eval=torch.device(device))
                
                # Randomly make batch_size * p trials incorrect
                random.seed((repeats+1)*(i+1)) # sets a different random seed for each repeat and batch but keeps seed consistent across epochs
                num_incorrect_trials = int(batch_size * p)
                selected_indices = random.sample(range(len(reward)), num_incorrect_trials)
                reward[selected_indices] = 1-reward[selected_indices]
                
                optimizer.zero_grad()
                
                assert predictions.device == reward.device, "Predictions and reward are on different devices"
                loss = criterion(predictions.to(device), reward.to(device))
                
                # make sure model is in training mode and on the device
                model.train()
                assert next(model.parameters()).device == device, "Model is not on the expected device"
                assert model.training, "Model is not in training mode"
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct_predictions += test_acc
                
            # Calculate average loss and accuracy for the training set
            average_loss = total_loss / num_batches
            average_accuracy = correct_predictions / num_batches
            
            # Store training loss and accuracy across epochs for this fold in a list
            train_loss_history.append(average_loss)
            train_accuracy_history.append(average_accuracy)
            
            # Validation phase
            with torch.no_grad():
                val_reward, val_predictions, val_acc, _ = evaluate_reward_model(fold_valid_X, fold_valid_y, model, eval_mode = True, device_eval=torch.device(device))
                val_loss = criterion(val_predictions.to(device), val_reward.to(device))
                
                # Store validation loss and accuracy across epochs for this fold in a list
                val_loss_history.append(val_loss.item())
                val_accuracy_history.append(val_acc)
            
            # Early stopping if val acc doesn't improve for 5 epochs    
            if val_acc > best_val_acc:
                assert val_acc == val_accuracy_history[-1]
                best_val_acc = val_accuracy_history[-1]
                best_model_val_loss = val_loss_history[-1]
                current_train_loss = train_loss_history[-1]
                current_train_acc = train_accuracy_history[-1]
                best_fold_model[fold] = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping! No improvement in validation accuracy for", patience, "epochs.")
                    break

        # Print metrics for this fold:
        print(f"     Fold {fold+1} - Train Loss: {current_train_loss:.4f}, Train Accuracy: {current_train_acc:.4f}")
        print(f"     Fold {fold+1} - Val Loss: {best_model_val_loss:.4f}, Val Accuracy: {best_val_acc:.4f}")
        
        fold_metrics[fold]['Training Loss'] = train_loss_history
        fold_metrics[fold]['Training Accuracy'] = train_accuracy_history
        fold_metrics[fold]['Validation Accuracy'] = val_accuracy_history
        fold_metrics[fold]['Validation Loss'] = val_loss_history
        
    return best_fold_model, fold_metrics

######################### Train logistic regression to predict reward from neural data
# Repeat the process for a number of iterations
def train_logR_reward(incorrect_feedback, correct_feedback, n_targ = 2, iterations = 100, n_components = 0.92):

    cm_pre_avg = np.zeros((n_targ, n_targ, iterations))
    accuracy_history = []

    for i in range(iterations):
        print('Iteration: ', i)
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
        pca = PCA(n_components = n_components, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Train a logistic regression model
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train_pca, y_train)

        # Make predictions and calculate accuracy
        predictions = model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_history.append(accuracy)

        cm = confusion_matrix(y_test, predictions)
        cm_pre_avg[:, :, i] = cm
    
    return cm_pre_avg, accuracy_history
    
    

