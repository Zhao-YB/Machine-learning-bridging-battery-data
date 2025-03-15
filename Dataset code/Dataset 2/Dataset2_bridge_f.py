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

import pickle
with open('u_charge_eis_input.pkl', 'rb') as f:
    u_charge_eis_input = pickle.load(f)

all_key_freqs = []
for df in u_charge_eis_input:
    group_df = df.iloc[4*16:5*16,:].head(6)

    kmeans = KMeans(n_clusters=2, init='k-means++')  
    kmeans.fit(group_df[['freq_Hz', 'z_im_comp_mOhm']])
    cluster_labels = kmeans.labels_

    key_freqs = []
    for j in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == j)[0]
        min_index = np.argmin(np.abs(group_df['freq_Hz'].iloc[cluster_indices] - np.mean(group_df['freq_Hz'].iloc[cluster_indices])))
        key_freq = group_df['freq_Hz'].iloc[cluster_indices[min_index]]
        key_freqs.append(key_freq)

    all_key_freqs.append(key_freqs)

all_key_freqs_tuples = [tuple(row) for row in all_key_freqs]
freq_counts = Counter(all_key_freqs_tuples)
most_common_freqs = freq_counts.most_common(1)[0][0]
print(most_common_freqs)

all_key_freqs = []
for df in u_charge_eis_input:
    group_df = df.iloc[4*16:5*16,:].tail(10)

    kmeans = KMeans(n_clusters=2, init='k-means++')  
    kmeans.fit(group_df[['freq_Hz', 'z_im_comp_mOhm']])
    cluster_labels = kmeans.labels_

    key_freqs = []
    for j in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == j)[0]
        min_index = np.argmin(np.abs(group_df['freq_Hz'].iloc[cluster_indices] - np.mean(group_df['freq_Hz'].iloc[cluster_indices])))
        key_freq = group_df['freq_Hz'].iloc[cluster_indices[min_index]]
        key_freqs.append(key_freq)

    all_key_freqs.append(key_freqs)

all_key_freqs_tuples = [tuple(row) for row in all_key_freqs]
freq_counts = Counter(all_key_freqs_tuples)
most_common_freqs = freq_counts.most_common(1)[0][0]
print(most_common_freqs)

with open('filtered_eis_dfs.pkl', 'rb') as f:
    filtered_eis_dfs = pickle.load(f)

    freqs_of_interest = [312.5, 10.0]

train_data = []
test_data = []


for i, df in enumerate(filtered_eis_dfs):
    df_soc_90 = df[df['soc_nom'] == 90]
    if df_soc_90.empty:
        continue

    df_filtered = df_soc_90[df_soc_90['freq_Hz'].isin(freqs_of_interest)]
    if df_filtered.empty:
        continue

    if len(df_filtered) >= 2:
        # Check for max z_re_comp_mOhm before splitting
        if df_filtered['z_re_comp_mOhm'].max() > 30:
            continue  # Skip this DataFrame

        split_index = len(df_filtered) // 2

        X = df_filtered.iloc[split_index:][['z_re_comp_mOhm', 'z_im_comp_mOhm', 't_avg_degC', 'SOC']].values
        y = df_filtered.iloc[:split_index][['z_re_comp_mOhm', 'z_im_comp_mOhm']].values

        if i % 3 != 2:  # Every 3rd DataFrame goes to test set, rest to train set
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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

real_intervals = [(14, 18), (18, 22), (22, 26), (26, 30)]

rf_models_x = [{} for _ in range(2)]   
rf_models_y = [{} for _ in range(2)]  
predictions_list = []

for set_idx in range(2):  
    X_train_set = X_train_sets[set_idx]
    y_train_set = y_train_sets[set_idx]
    
    for real_start, real_end in real_intervals:
            mask = (
                (X_train_set[:, 0] >= real_start) & (X_train_set[:, 0] < real_end) 
            )
            
            X_train_subset = X_train_set[mask]
            y_train_subset = y_train_set[mask]
            
            if X_train_subset.shape[0] == 0:
                continue

            y_train_x = y_train_subset[:, 0]  
            y_train_y = y_train_subset[:, 1]  
            
            X_train_x_features = X_train_subset[:, [0, 2]]  
            X_train_y_features = X_train_subset[:, [1, 2]]  

            rf_model_x = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model_x.fit(X_train_x_features, y_train_x)
            rf_models_x[set_idx][(real_start, real_end)] = rf_model_x
            
            rf_model_y = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model_y.fit(X_train_y_features, y_train_y)
            rf_models_y[set_idx][(real_start, real_end)] = rf_model_y

for set_idx in range(2):  
    X_test_set = X_test_sets[set_idx]
    y_test_set = y_test_sets[set_idx]
    y_test_x = y_test_set[:, 0]
    y_test_y = y_test_set[:, 1]
    
    for idx, test_sample in enumerate(X_test_set):
        real_part = test_sample[0]
        imag_part = test_sample[1]

        selected_interval = None
        for (real_start, real_end, imag_start, imag_end) in rf_models_x[set_idx].keys():
            if real_start <= real_part < real_end and imag_start <= imag_part < imag_end:
                selected_interval = (real_start, real_end, imag_start, imag_end)
                break
        
        if selected_interval:
            rf_model_x = rf_models_x[set_idx][selected_interval]
            rf_model_y = rf_models_y[set_idx][selected_interval]
            
            X_test_x_features = test_sample[[0, 2]].reshape(1, -1)
            X_test_y_features = test_sample[[1, 2]].reshape(1, -1)

            rf_prediction_x = rf_model_x.predict(X_test_x_features)[0]
            rf_prediction_y = rf_model_y.predict(X_test_y_features)[0]

            predictions_dict = {
                'set': set_idx + 1,  
                'sample_idx': idx,
                'real_interval': selected_interval[:2],
                'imag_interval': selected_interval[2:],
                'predicted_x': rf_prediction_x,
                'predicted_y': rf_prediction_y,
                'true_x': y_test_x[idx],
                'true_y': y_test_y[idx]
            }
            predictions_list.append(predictions_dict)

predictions_df = pd.DataFrame(predictions_list)

mid_freq_end = 16

key_freqs = (312.5, 10.0)


all_input_data_key = []
all_output_data_high = []
all_input_data_high_to_mid = []
all_output_data_mid = []
all_input_data_high_mid_to_low = []
all_output_data_low = []

for df in u_charge_eis_input:
    df_subset = df.iloc[4*16:5*16,:].values
    input_key = []
    for freq in key_freqs:
        dff = df_subset[df_subset[:, 0] == freq]
        if len(dff) > 0:
            input_key.append(dff[0, 1])
    input_key = np.array(input_key)
    
    output_high = df_subset[0:mid_freq_end, 1:3]  


    all_input_data_key.append(input_key)
    all_output_data_high.append(output_high)

    
all_input_data_key = np.array(all_input_data_key)
all_output_data_high = np.array(all_output_data_high)

train_input_key, test_input_key, train_output_high, test_output_high = train_test_split(
    all_input_data_key, all_output_data_high, test_size=0.2, random_state=42
)


scaler_key = StandardScaler()
scaler_high = StandardScaler()
scaler_mid = StandardScaler()
scaler_low = StandardScaler()
scaler_high_to_mid = StandardScaler()
scaler_high_mid_to_low = StandardScaler()

train_input_key_scaled = scaler_key.fit_transform(train_input_key.reshape(train_input_key.shape[0], -1))
train_output_high_scaled = scaler_high.fit_transform(train_output_high[:,:,0].reshape(train_output_high[:,:,0].shape[0], -1))

test_input_key_scaled = scaler_key.transform(test_input_key.reshape(test_input_key.shape[0], -1))

model_key_to_high_rf = RandomForestRegressor(n_estimators=100,
                                             criterion='mse',
                                             max_depth=10,
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             min_weight_fraction_leaf=0.0,
                                             max_features='auto',
                                             max_leaf_nodes=None,
                                             bootstrap=True,
                                             oob_score=False,
                                             n_jobs=1,
                                             random_state=42,
                                             verbose=0,
                                             warm_start=False)
model_key_to_high_rf.fit(train_input_key_scaled, train_output_high_scaled)

num_dfs = len(u_charge_eis_input)
input_data = np.zeros((num_dfs, 16, 1))

for i, df in enumerate(u_charge_eis_input):

  data = df[['z_re_comp_mOhm']].values

  data = data[4*16:5*16, :]

  input_data[i, :, :] = data
    
train_input, test_input, train_output , test_output = train_test_split(
    input_data, u_charge_output, test_size=0.2, random_state=42  
)
train_input_flattened = train_input.reshape(train_input.shape[0], -1)  
test_input_flattened = test_input.reshape(test_input.shape[0], -1)     

scaler_input = MinMaxScaler()
train_input_normalized = scaler_input.fit_transform(train_input_flattened)
test_input_normalized = scaler_input.transform(test_input_flattened)

scaler_output = MinMaxScaler()
train_output_normalized = scaler_output.fit_transform(train_output)
test_output_normalized = scaler_output.transform(test_output)

rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(train_input_normalized, train_output_normalized)
rf_predictions = rf_model.predict(test_input_normalized)
with open('u_discharge_eis_input.pkl', 'rb') as f:
    u_discharge_eis_input = pickle.load(f)

u_discharge_output = np.load("u_discharge_output.npy")
num_dfs = len(u_discharge_eis_input)
input_data = np.zeros((num_dfs, 16, 1))

for i, df in enumerate(u_discharge_eis_input):
  data = df[['z_re_comp_mOhm']].values

  data = data[4*16:5*16, :]

  input_data[i, :, :] = data
    
train_input, test_input, train_output , test_output = train_test_split(
    input_data, u_discharge_output, test_size=0.2, random_state=42  
)
train_input_flattened = train_input.reshape(train_input.shape[0], -1)  
test_input_flattened = test_input.reshape(test_input.shape[0], -1)    

scaler_input = MinMaxScaler()
train_input_normalized = scaler_input.fit_transform(train_input_flattened)
test_input_normalized = scaler_input.transform(test_input_flattened)

scaler_output = MinMaxScaler()
train_output_normalized = scaler_output.fit_transform(train_output)
test_output_normalized = scaler_output.transform(test_output)