import pandas as pd  
import matplotlib.pyplot as plt  
%matplotlib inline
import numpy as np
import config_main as cfg
import config_labels as csv_label
import os
import pandas as pd
import numpy as np

min_length = 100
eis_input_dir = "G:\\Project\\Comprehensive battery aging dataset\\eisdata\\data\\dataset\\cell_eisv2"
csv_input_dir = "G:\\Project\\Comprehensive battery aging dataset\\cell_log_age\\data\\dataset"

eis_files = [f for f in os.listdir(eis_input_dir) if f.endswith('.csv')]
csv_files = [f for f in os.listdir(csv_input_dir) if f.endswith('.csv')]
segmented_eis_dfs = []  
segmented_csv_dfs = []  

soc_sequence = [10, 30, 50, 70, 90, 90, 70, 50, 30, 10]  
group_size = len(soc_sequence) * 29  

def get_reference_freq(df):
    return df.iloc[:29]['freq_Hz'].values

def process_eis_data(eis_df, reference_freq):
    result = []
    i = 0
    while i + group_size <= len(eis_df):
        group = eis_df.iloc[i:i + group_size]
        soc_values = group['soc_nom'].values.reshape(-1, 29)[:, 0]
        freq_values = group['freq_Hz'].values.reshape(-1, 29)
        
        if np.array_equal(soc_values, soc_sequence) and np.all(freq_values == reference_freq):
            result.append(group)
            i += group_size
        else:
            i += 1
    return result

def process_csv_data(csv_df):
    delta_q_Ah = csv_df['delta_q_Ah'].values
    non_zero_segments = []
    non_zero_indices = np.where(delta_q_Ah != 0)[0]
    if len(non_zero_indices) == 0:
        return []
    else:
        start_index = non_zero_indices[0]
        for i in range(1, len(non_zero_indices)):
            if non_zero_indices[i] - non_zero_indices[i - 1] > min_length:
                segment = csv_df['delta_q_Ah'][start_index:non_zero_indices[i - 1] + 1]
                if len(segment) < 2000 and np.sum(segment != 0) >= min_length and segment.values.max() > 1 and segment.values.min() < -1:
                    non_zero_segments.append(segment)
                start_index = non_zero_indices[i]

        segment = csv_df['delta_q_Ah'][start_index:non_zero_indices[-1] + 1]
        if len(segment) < 2000 and np.sum(segment != 0) >= min_length and segment.values.max() > 1 and segment.values.min() < -1:
            non_zero_segments.append(segment)
    
    return non_zero_segments


for eis_file, csv_file in zip(eis_files, csv_files):
    print(f"processing£ºEIS: {eis_file} | CSV: {csv_file}")
    
    eis_df = pd.read_csv(os.path.join(eis_input_dir, eis_file), header=0, sep=cfg.CSV_SEP)
    csv_df = pd.read_csv(os.path.join(csv_input_dir, csv_file), header=0, sep=cfg.CSV_SEP)

    reference_freq = get_reference_freq(eis_df)
    eis_segments = process_eis_data(eis_df, reference_freq)

    csv_segments = process_csv_data(csv_df)

    if len(eis_segments) >= len(csv_segments):
        for eis_segment, csv_segment in zip(eis_segments[:len(csv_segments)], csv_segments):
            if len(eis_segment) == 290:
                start_index_eis = eis_segment.index[0]
                end_index_eis = eis_segment.index[-1]
                segmented_eis_df = eis_df.iloc[start_index_eis:end_index_eis + 1]
                segmented_eis_dfs.append(segmented_eis_df)
                start_index_csv = csv_segment.index[0]
                end_index_csv = csv_segment.index[-1]
                segmented_csv_df = csv_df.iloc[start_index_csv:end_index_csv + 1]
                segmented_csv_dfs.append(segmented_csv_df)
    else:
        print(f"file {csv_file} skip")


empty_value_frequencies = {}

for i in range(len(segmented_eis_dfs)):
    eis_segment = segmented_eis_dfs[i][['freq_Hz', 'z_re_comp_mOhm', 'z_im_comp_mOhm', 'soc_nom', 't_avg_degC']]
    
    eis_segment_filtered = eis_segment[(eis_segment['freq_Hz'] <= 1000) & (eis_segment['freq_Hz'] >= 2)]
    
    null_rows = eis_segment_filtered[eis_segment_filtered.isnull().any(axis=1)]
    
    if not null_rows.empty:
        empty_value_frequencies[i] = null_rows['freq_Hz'].values
        print(f"index {i} Nan£¬Freq is : {null_rows['freq_Hz'].values}")


filtered_eis_dfs = []
filtered_csv_dfs = []

for i in range(len(segmented_eis_dfs)):
    eis_segment = segmented_eis_dfs[i][['freq_Hz', 'z_re_comp_mOhm', 'z_im_comp_mOhm', 'soc_nom', 't_avg_degC']]
    
    eis_segment_filtered = eis_segment[(eis_segment['freq_Hz'] <= 1000) & (eis_segment['freq_Hz'] >= 2)]
    
    eis_segment_filtered = eis_segment_filtered[eis_segment_filtered['freq_Hz'] != 100]
    
    if not eis_segment_filtered.isnull().values.any():
        filtered_eis_dfs.append(eis_segment_filtered)
        filtered_csv_dfs.append(segmented_csv_dfs[i])
    else:
        null_rows = eis_segment_filtered[eis_segment_filtered.isnull().any(axis=1)]
        print(f"delete index {i} Nan")

filtered_segment = []
for segments in filtered_csv_dfs:
    data = segments.delta_q_Ah.values

    min_value_index = np.argmin(data)
    max_value_index = np.argmax(data)

    if min_value_index > max_value_index:
        start_index = max_value_index
        for i in range(max_value_index - 1, -1, -1):
            if data[i] >= data[i + 1]:  
                start_index = i + 1  
                break

        filtered_segment.append(segments.iloc[start_index:min_value_index + 1])
    else:
        
        min_value_after_max = np.min(data[max_value_index+1:])  
        min_value_after_max_index = np.where(data == min_value_after_max)[0][-1]  

        start_index = max_value_index
        for i in range(max_value_index - 1, -1, -1):
            if data[i] >= data[i + 1]:  
                start_index = i + 1  
                break

        filtered_segment.append(segments.iloc[start_index:min_value_after_max_index + 1])
from scipy.interpolate import interp1d

u_charge_output = []
error_indices = []  

for i in range(len(filtered_segment)):
    try:
        dff = filtered_segment[i][(filtered_segment[i].i_raw_A < 1.1) & (filtered_segment[i].i_raw_A > 0.9)]

        dff = dff.sort_values(by="v_raw_V")

        f = interp1d(dff["v_raw_V"], dff["delta_q_Ah"], kind='linear',
                     fill_value="extrapolate") 
        new_voltage = np.arange(2.5, 4.1, 0.01)  
        new_q = f(new_voltage)  
        u_charge_output.append(new_q)
    except ValueError:
        print(f"index {i}  ValueError :")
        error_indices.append(i)  

print(f" ValueError index£º{error_indices}")
u_charge_output = np.array(u_charge_output)

u_discharge_output = []
error_indices = []  

for i in range(len(filtered_segment)):
    try:
        dff = filtered_segment[i][(filtered_segment[i].i_raw_A > -1.1) & (filtered_segment[i].i_raw_A < -0.9)]
        dff = dff.sort_values(by="v_raw_V")

        f = interp1d(dff["v_raw_V"], dff["delta_q_Ah"], kind='linear',
                     fill_value="extrapolate")  
        new_voltage = np.arange(2.5, 4.1, 0.01)  
        new_q = f(new_voltage)  
        u_discharge_output.append(new_q)
    except ValueError:
        print(f"index {i}  ValueError :")
        error_indices.append(i)  

print(f" ValueError £º{error_indices}")
u_discharge_output = np.array(u_discharge_output)

from scipy.interpolate import interp1d
i_discharge_output = []
error_indices = []  

for i in range(len(filtered_segment)):
    try:
        dff = filtered_segment[i].iloc[2:,:][(filtered_segment[i].iloc[2:,:].i_raw_A < 0.9) & (filtered_segment[i].iloc[2:,:].i_raw_A > 0.1)]
        dff = dff.sort_values(by="i_raw_A")

        f = interp1d(dff["i_raw_A"], dff["delta_q_Ah"], kind='linear',
                     fill_value="extrapolate")  
        new_current = np.arange(0.1, 0.8, 0.01) 
        new_q = f(new_current)  
        i_discharge_output.append(new_q)
    except ValueError:
        print(f"index {i}  ValueError :")
        error_indices.append(i)  

print(f" ValueError {error_indices}")
i_discharge_output = np.array(i_discharge_output)

import numpy as np
from scipy.interpolate import interp1d

i_charge_output = []
error_indices = []  
large_diff_indices = []  

for i in range(len(filtered_segment)):
    try:
        dff = filtered_segment[i].iloc[filtered_segment[i].i_raw_A.argmin():,:]
        dff = dff.sort_values(by="i_raw_A")


        min_value = np.min(dff.i_raw_A)
        min_indices = np.where(dff.i_raw_A == min_value)[0]
        last_min_index = min_indices[-1]  
    
        dff = dff.iloc[last_min_index:, :]

        f = interp1d(dff["i_raw_A"], dff["delta_q_Ah"], kind='linear',
                     fill_value="extrapolate")  
        new_current = np.arange(-0.7, -0.1, 0.01)
        new_q = f(new_current)  

        if np.max(new_q) - np.min(new_q) > 0.5:
            large_diff_indices.append(i)
        else:  
            i_charge_output.append(new_q)

    except ValueError:
        print(f"index {i}  ValueError :")
        error_indices.append(i)  


