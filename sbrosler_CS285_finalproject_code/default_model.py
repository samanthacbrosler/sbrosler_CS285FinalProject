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


class CnnRnn(torch.nn.Module): 
    """
    The CNN RNN classifier from Generalizable Spelling paper. 
    """
    def __init__(self, rnn_dim=253, KS=4, num_layers=2, dropout=0.54, n_targ=30, bidirectional=True, in_channels=253):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x): 
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
        # x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        _ , x = self.BiGRU(x)
        x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
        x = x[-1] # Only care about the output at the final layer.
        # (2, bs, rnn_dim)
        
        x= x.contiguous().permute(1, 0, 2)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.dense(x)
        return out 