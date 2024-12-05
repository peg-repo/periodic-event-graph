import sys
sys.path.append('./lib/pot/src/')
sys.path.append('./lib/mass/')         
sys.path.append('./lib/DyGLib/')

import pandas as pd
import numpy as np
import matrixprofile as mp
from fastdtw import fastdtw
import hdbscan
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft
from spot import spot
from pot import pot
from spot_sy import spot_sy
from mass_ts._mass_ts_sy import mass2 as mass2_sy

def extract_residual_windows_df(window_df, event_set, pattern_matching_df):
    residual_windows_df = pd.DataFrame(index = window_df.index, columns =  window_df.columns)

    for col_name in residual_windows_df.columns:
        for i in residual_windows_df.index:
            event_key = pattern_matching_df[col_name][i]
            # Apply padding to adjust lengths when they differ
            if len(window_df[col_name][i]) < len(event_set[event_key]):
                padded_window = np.pad(window_df[col_name][i], (0, len(event_set[event_key]) - len(window_df[col_name][i])), 'constant')
                residual_windows_df[col_name][i] = padded_window - event_set[event_key]
            elif len(window_df[col_name][i]) > len(event_set[event_key]):
                padded_event = np.pad(event_set[event_key], (0, len(event_set[event_key]) - len(window_df[col_name][i])), 'constant')
                residual_windows_df[col_name][i] = window_df[col_name][i] - padded_event
            else:
                residual_windows_df[col_name][i] = window_df[col_name][i] - event_set[event_key]

    return residual_windows_df

def extract_threshold_windows_df(df, data_dict, size, stride, init_level = 0.8):
    threshold_df = pd.DataFrame(index = df.index, columns = df.columns)

    for col_name in df.columns:
        if col_name in ('AC4', 'Refrigerator'):
            threshold_df[col_name] = [max(data_dict[col_name])] * len(data_dict[col_name])
        else:
            init = int(len(df)*0.8)
            threshold = spot_sy(data_dict[col_name], num_init = init, init_level = init_level, risk = 1e-4)
            threshold_df[col_name] = threshold['t']
            
    num_windows = (len(threshold_df) - size) // stride + 1
    range_windows = range(0, num_windows * stride, stride)
    threshold_windows_df = pd.DataFrame(index=range_windows, columns=threshold_df.columns)

    for col_name in threshold_df.columns:
        windows = []
        window_size = size
        for i in range_windows:
            data = threshold_df[col_name][i : i + window_size].values
            windows.append(data.tolist())
        threshold_windows_df[col_name] = windows
        
    return threshold_windows_df

def residual_matching_df(residual_window_df, threshold_window_df):
    residual_matching_df = pd.DataFrame(index = residual_window_df.index, columns =  residual_window_df.columns)

    for col_name in residual_matching_df.columns:
        for i in residual_matching_df.index:
            
            if len(residual_window_df[col_name][i]) < len(threshold_window_df[col_name][i]):
                padded_residual = np.pad(residual_window_df[col_name][i],
                                         (0, len(threshold_window_df[col_name][i]) - len(residual_window_df[col_name][i])),
                                         'constant')
                condition = np.any((padded_residual - threshold_window_df[col_name][i]) > 0)
            elif len(residual_window_df[col_name][i]) > len(threshold_window_df[col_name][i]):
                padded_threshold = np.pad(threshold_window_df[col_name][i],
                                         (0, len(residual_window_df[col_name][i]) - len(threshold_window_df[col_name][i])),
                                         'constant')
                condition = np.any((residual_window_df[col_name][i] - padded_threshold) > 0)
            else:
                condition = np.any((residual_window_df[col_name][i] - threshold_window_df[col_name][i]) > 0) 

            if condition == True:
                residual_matching_df[col_name][i] = 'Residual +'
            else:
                residual_matching_df[col_name][i] = 'Residual -'
    
    return residual_matching_df
    
def event2graph_df(pattern_df, residual_df, evnet_set):
    time_series_node = list(pattern_df.columns)
    event_node = list(evnet_set.keys()) + ['Residual +', 'Residual -']  # residual error event node
    
    timeseries_mapping = {time_series: idx for idx, time_series in enumerate(time_series_node)}
    event_mapping = {event: idx for idx, event in enumerate(event_node)}
    
    pattern_matching_mapped = pattern_df.copy() 
    for column in pattern_df.columns:
        pattern_matching_mapped[column] = pattern_df[column].map(event_mapping)
    
    residual_matching_mapped = residual_df.copy() 
    for column in residual_df.columns:
        residual_matching_mapped[column] = residual_df[column].map(event_mapping)

    event_df = pd.melt(pattern_matching_mapped.reset_index(), id_vars = ['index'], var_name = 'u', value_name = 'i').rename(columns={'index': 'ts'})
    residual_df = pd.melt(residual_matching_mapped.reset_index(), id_vars = ['index'], var_name = 'u', value_name = 'i').rename(columns={'index': 'ts'})

    graph_df = pd.concat([event_df, residual_df], axis=0)
    graph_df = graph_df.reset_index()
    graph_df['label'] = 0.0  #state level
    graph_df = graph_df[['u', 'i', 'ts', 'label']]
    graph_df['u'] = graph_df['u'].map(timeseries_mapping)   # appiance mapping

    edge_features_df = pd.concat([pd.get_dummies(graph_df['u'], prefix='u', dtype=int),
                                  pd.get_dummies(graph_df['i'], prefix='i', dtype=int)], axis=1)
    graph_df = pd.concat([graph_df, edge_features_df], axis=1)
    graph_df = graph_df.sort_values(by=['ts', 'u'])
    graph_df = graph_df.reset_index(drop=True)

    return graph_df
