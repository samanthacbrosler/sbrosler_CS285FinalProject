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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import gaussian_filter1d



def extract_data(erp_collection):
    
    # Takes in ERP collection and returns np.array in the shape of trials x time x channels
    
    X = []
    y = []
    
    # Build feature (X) and label (y) vectors from the ERP collection
    for group_name, erp_group in erp_collection.items():
        
        # Ensure erp_group is a list
        if not isinstance(erp_group, list):
            erp_group = [erp_group]
            
        # Collect ERPs
        group_erps = np.vstack(np.array([erp_set['erps'] for erp_set in erp_group]))
        
        # Collect event labels
        group_labels = np.hstack(np.array([erp_set['event_label_texts'] for erp_set in erp_group]))

        X.append(group_erps)
        y.append(group_labels)

    # Flatten across groups
    X = np.vstack(np.array(X))
    y = np.hstack(np.array(y))
    
    # Find the unique elements in the label y and map them to numbers
    unique_elements, element_numbers = np.unique(y, return_inverse=True)

    # Convert the list to a NumPy array with mapped numbers
    y_num = np.array(element_numbers)
    
    print('X shape:', X.shape)
    print('y shape:', y.shape)
  
    return X, y, y_num


def preprocess(X, y, y_num, downsampling_factor):
    
    print('Original data shape (trials x time x channels): ', X.shape)
    
    # Apply decimation
    print('\nDecimating the time series dimension...')
    downsampled_X = signal.decimate(X, downsampling_factor, axis=1)
    print('Downsampled data shape: ', downsampled_X.shape)
    
    # Normalize across channels at each time point using L2 norm
    print('\nNormalizing data across channels at each time point...')
    normalized_data = normalize(downsampled_X)
    
    # Shuffle data
    print('\nShuffling trial order...')
    X, y, y_num = shuffle_data(normalized_data, y, y_num)
    
    print('\nFinished preprocessing!')
    
    return X, y, y_num


def normalize(x, axis=-1, order=2):
    """
    This is from the keras source code https://github.com/keras-team/keras/blob/v2.7.0/keras/utils/np_utils.py#L77-L91
    
    Normalizes a Numpy array.
    Args:
      x: Numpy array to normalize.
      axis: axis along which to normalize.
      order: Normalization order (e.g. `order=2` for L2 norm).
    Returns:
      A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


def shuffle_data(X, y, y_num):
    
    np.random.seed(42)
    shuffled_indices = np.random.permutation(X.shape[0])
    X_shuffle = X[shuffled_indices]
    y_shuffle = y[shuffled_indices]
    y_num_shuffle = y_num[shuffled_indices]
    
    return X_shuffle, y_shuffle, y_num_shuffle


def feature_scaling(features):
    
    # Where "features" is a matrix in the shape of trials x channels x features
    
    # Reshape the matrix of features to be trials x (channels * features)
    X = features.reshape(features.shape[0], -1)

    # Create a StandardScaler object and fit it to the data
    scaler = StandardScaler()
    scaler.fit(X)

    # Scale the data using the scaler
    X_scaled = scaler.transform(X)

    # Reshape the scaled matrix back to the original shape of trials x channels x features
    X_scaled = X_scaled.reshape(features.shape)
    
    return X_scaled


def area_feature(data):
    
    area = trapz(data, axis=1)
    #features_winner = feature_scaling(area)
    
    return area