import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 300 
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.pylab import rcParams
from scipy.stats import pearsonr, spearmanr
import scipy.io as sio
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn import cluster
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
np.random.seed(42)

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC, LinearSVC, NuSVC

from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, OneSidedSelection
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from imblearn.datasets import make_imbalance
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.cluster import KMeans
import time
import h5py
from pyDRTtools.runs import simple_run, EIS_object,Bayesian_run,peak_analysis
from pyDRTtools.GUI import Figure_Canvas
from pyDRTtools import basics, HMC, BHT,GUI
from tensorflow.keras.models import load_model
import pyDRTtools
from pyDRTtools.runs import simple_run, EIS_object,Bayesian_run,peak_analysis
from pyDRTtools.GUI import Figure_Canvas
from pyDRTtools import basics, HMC, BHT,GUI
from math import sqrt, log
import numpy as np
from numpy.linalg import norm
from numpy import inf, sin, cos, cosh, pi, exp, log10
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import time
import os

folder_path = "../data"
file_prefix_ocv = "OCV_wExpansion"
extracted_ocvdata_dict = {}

for i in range(1, 22):
   
    folder_num = str(i).zfill(2)  
    ocv_filename = f"{file_prefix_ocv}.csv"

    ocv_filepath = os.path.join(folder_path, folder_num, ocv_filename)

    data_ocv = pd.read_csv(ocv_filepath)

    extracted_ocvdata = data_ocv[
        ["Time [s]", "Current [mA]", "Voltage [V]", "Capacity [Ah]", "Cycle number"]
    ]

    extracted_ocvdata_dict[folder_num] = extracted_ocvdata

folder_path = "../data"
file_prefix_resistance = "Resistance"

extracted_redata_dict = {}

for i in range(1, 22):
    folder_num = str(i).zfill(2)  
    resistance_filename = f"{file_prefix_resistance}.csv"
    resistance_filepath = os.path.join(folder_path, folder_num, resistance_filename)

    resistance_data = pd.read_csv(resistance_filepath)

    filtered_data = resistance_data[resistance_data["Re [Ohm]"] != 0]

    extracted_redata = filtered_data[
        ["Frequency [Hz]", "Re [Ohm]", "-Im [Ohm]", "Cycle number", 'Temperature [C]']
    ]
    

    extracted_redata["Re [Ohm]"] = extracted_redata["Re [Ohm]"] * 1000
    extracted_redata["-Im [Ohm]"] = extracted_redata["-Im [Ohm]"] * 1000

    extracted_redata_dict[folder_num] = extracted_redata

freqs_of_interest = [193.03522, 5.5343843]

train_data = []
test_data = []

for file_key in range(1, 22):
    file_key_str = str(file_key).zfill(2)  
    extracted_redata = extracted_redata_dict[file_key_str]
    extracted_ocvdata = extracted_ocvdata_dict[file_key_str]  
    unique_cycles = extracted_redata["Cycle number"].unique()

    for cycle in unique_cycles:
        cycle_data = extracted_redata[extracted_redata["Cycle number"] == cycle]
        first_36_rows = cycle_data.head(36)
        Field = cycle_data.iloc[36:72]
        
        df_soc_90 = first_36_rows
        if df_soc_90.empty:
            continue

        df_filtered_X = df_soc_90[df_soc_90['Frequency [Hz]'].isin(freqs_of_interest)]
        df_filtered_y = Field[Field['Frequency [Hz]'].isin(freqs_of_interest)]

        X = df_filtered_X[['Re [Ohm]', '-Im [Ohm]', 'Temperature [C]']].values
        y = df_filtered_y[['Re [Ohm]', '-Im [Ohm]']].values

        if file_key % 3 != 2:  # Every 3rd DataFrame goes to test set, rest to train set
            train_data.append((X, y))
        else:
            test_data.append((X, y))

# Combine the training data
X_train = np.concatenate([X for X, _ in train_data])
y_train = np.concatenate([y for _, y in train_data])

# Combine the test data
X_test = np.concatenate([X for X, _ in test_data])
y_test = np.concatenate([y for _, y in test_data])

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

X_train_sets = [[] for _ in range(2)]
y_train_sets = [[] for _ in range(2)]
X_test_sets = [[] for _ in range(2)]
y_test_sets = [[] for _ in range(2)]


for i in range(len(X_train)):
    set_index = i % 2  # Determine set index (0, 1, or 2)
    X_train_sets[set_index].append(X_train[i])
    y_train_sets[set_index].append(y_train[i])

for i in range(len(X_test)):
    set_index = i % 2  # Determine set index (0, 1, or 2)
    X_test_sets[set_index].append(X_test[i])
    y_test_sets[set_index].append(y_test[i])


# Convert lists of lists to NumPy arrays
X_train_sets = [np.array(s) for s in X_train_sets]
y_train_sets = [np.array(s) for s in y_train_sets]
X_test_sets = [np.array(s) for s in X_test_sets]
y_test_sets = [np.array(s) for s in y_test_sets]


for j in range(2):
    print(f"X_train set {j+1} shape:", X_train_sets[j].shape)
    print(f"y_train set {j+1} shape:", y_train_sets[j].shape)
    print(f"X_test set {j+1} shape:", X_test_sets[j].shape)
    print(f"y_test set {j+1} shape:", y_test_sets[j].shape)

# Define interval mapping
real_intervals = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16)]
imag_intervals = [(0, 0.5), (0.5, 2), (2, 4)]

# Initialize model storage structure for train set 0 and train set 1
rf_models_x = [{} for _ in range(2)]  # Models for real part x-coordinate, divided into 2 subsets
rf_models_y = [{} for _ in range(2)]  # Models for real part y-coordinate, divided into 2 subsets
predictions_list = []

# Train
# Train models for each training subset
for set_idx in range(2):  # Only 2 subsets, 0 and 1
    X_train_set = X_train_sets[set_idx]
    y_train_set = y_train_sets[set_idx]
    
    # Train for each real and imaginary interval
    for real_start, real_end in real_intervals:
        for imag_start, imag_end in imag_intervals:
            
            # Filter training data based on intervals
            mask = (
                (X_train_set[:, 0] >= real_start) & (X_train_set[:, 0] < real_end) &
                (X_train_set[:, 1] >= imag_start) & (X_train_set[:, 1] < imag_end)
            )
            
            # Select training data and labels that meet the conditions
            X_train_subset = X_train_set[mask]
            y_train_subset = y_train_set[mask]
            
            # Skip if no data points meet the conditions
            if X_train_subset.shape[0] == 0:
                continue

            # Get target values for x and y coordinates
            y_train_x = y_train_subset[:, 0]  # x-coordinate
            y_train_y = y_train_subset[:, 1]  # y-coordinate
            
            # Extract features for x and y models
            X_train_x_features = X_train_subset[:, [0, 2]]  # Use columns 0, 2 as x features
            X_train_y_features = X_train_subset[:, [1, 2]]  # Use columns 1, 2 as y features

            # Train and save the x-coordinate model
            rf_model_x = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model_x.fit(X_train_x_features, y_train_x)
            rf_models_x[set_idx][(real_start, real_end, imag_start, imag_end)] = rf_model_x
            
            # Train and save the y-coordinate model
            rf_model_y = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model_y.fit(X_train_y_features, y_train_y)
            rf_models_y[set_idx][(real_start, real_end, imag_start, imag_end)] = rf_model_y

# Testing phase: Ensure training and test sets are fully aligned
for set_idx in range(2):  # For each test subset
    X_test_set = X_test_sets[set_idx]
    y_test_set = y_test_sets[set_idx]
    y_test_x = y_test_set[:, 0]
    y_test_y = y_test_set[:, 1]
    
    # Select the appropriate model based on the real and imaginary parts of each test sample
    for idx, test_sample in enumerate(X_test_set):
        real_part = test_sample[0]
        imag_part = test_sample[1]

        # Find the appropriate interval
        selected_interval = None
        for (real_start, real_end, imag_start, imag_end) in rf_models_x[set_idx].keys():
            if real_start <= real_part < real_end and imag_start <= imag_part < imag_end:
                selected_interval = (real_start, real_end, imag_start, imag_end)
                break
        
        # If a matching interval is found, use the corresponding model for prediction
        if selected_interval:
            rf_model_x = rf_models_x[set_idx][selected_interval]
            rf_model_y = rf_models_y[set_idx][selected_interval]
            
            # Extract features for x and y models
            X_test_x_features = test_sample[[0, 2]].reshape(1, -1)
            X_test_y_features = test_sample[[1, 2]].reshape(1, -1)

            rf_prediction_x = rf_model_x.predict(X_test_x_features)[0]
            rf_prediction_y = rf_model_y.predict(X_test_y_features)[0]

            # Store prediction results and true values
            predictions_dict = {
                'set': set_idx + 1,  # Ensure subset index starts from 1
                'sample_idx': idx,
                'real_interval': selected_interval[:2],
                'imag_interval': selected_interval[2:],
                'predicted_x': rf_prediction_x,
                'predicted_y': rf_prediction_y,
                'true_x': y_test_x[idx],
                'true_y': y_test_y[idx]
            }
            predictions_list.append(predictions_dict)

# Convert results to DataFrame
predictions_df = pd.DataFrame(predictions_list)