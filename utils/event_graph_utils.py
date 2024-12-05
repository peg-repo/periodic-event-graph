import sys
sys.path.append('./lib/peak-over-threshold/src/')
sys.path.append('./lib/mass-ts/')         
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

def eg_matrix_profile_set(df, window_size, top_k):
    profiles_set = {}
    
    for col_name in df.columns:
        key = '{} Profile'.format(col_name)
        profiles_set[key] = mp.compute(df[col_name].values, window_size)
        profiles_set[key] = mp.discover.motifs(profiles_set[key],
                                               exclusion_zone = window_size//2,
                                               k = top_k)
            
    return profiles_set

def dtw_distance(x, y):
    distance, _ = fastdtw(x, y)
    return distance

def dtw_distance_matrix(time_series_list):
    num_series = len(time_series_list)
    distance_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in range(i, num_series):
            distance_matrix[i, j] = dtw_distance(time_series_list[i], time_series_list[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # 대칭 행렬
            
    return distance_matrix

def eg_event_generation_set(df, mp_set, window_size, cluster_size):
    motifs = {}
    for col_name in df.columns:
        key = '{} Profile'.format(col_name)
        index = [item['motifs'] for item in mp_set[key]['motifs']]                
        motifs[col_name] = list(set(sum(index, [])))                    # exclude duplicate indices

    patterns = []
    for col_name in df.columns:
        patterns.append([df[col_name].values[idx : idx + window_size] for idx in motifs[col_name]])
    patterns = list({tuple(array) for array in sum(patterns, [])})      # exclude duplicate
    patterns = [np.array(array) for array in patterns]


    hdb_cluster = hdbscan.HDBSCAN(metric = dtw_distance, min_cluster_size = cluster_size)
    hdb_cluster.fit(patterns)
    cls_labels = hdb_cluster.labels_
    
    events_list = []
    for i in np.unique(cls_labels)[np.unique(cls_labels) != -1]:
        events_list.append(hdb_cluster.weighted_cluster_centroid(cluster_id = i))
    #for i in list(np.where(cls_labels == -1)[0]):
    #    events_list.append(patterns[i])

    events_set = {}
    for i, event in enumerate(events_list):
        key = 'Event {}'.format(i + 1)
        events_set[key] = event 

    return events_set

def eg_extract_windows_df(df, window_size, window_stride):
    
    num_windows = (len(df) - window_size) // window_stride + 1
    range_windows = range(0, num_windows * window_stride, window_stride)
    windows_df = pd.DataFrame(index = range_windows, columns =  df.columns)

    windows = []
    for col_name in df.columns:
        windows = []
        for i in range_windows:
            data = df[col_name][i : i + window_size].values
            windows.append(data.tolist())
        windows_df[col_name] = windows

    return windows_df

def eg_pattern_matching_df(window_df, event_set):
    similarity_tensor = {}
    
    for i in window_df.index:
        sliding_window = window_df.loc[i]
        similarity = {}
        
        for col_name in window_df.columns:
            ts_similarity = []
            for event in event_set.values():
                ts_similarity.append(mass2_sy(sliding_window[col_name], event))
            similarity[col_name] = np.concatenate(ts_similarity)
            
        similarity_tensor[i] = similarity

    
    event_matching_df = pd.DataFrame(index = window_df.index, columns =  window_df.columns)
    
    for i in similarity_tensor.keys():
        similarity = similarity_tensor[i]
        for col_name in similarity.keys():
            event_matching_df[col_name][i] = 'Event {}'.format(np.nanargmin(similarity[col_name]) + 1)

    return event_matching_df

def eg_extract_residual_windows_df(window_df, event_set, pattern_matching_df):
    residual_windows_df = pd.DataFrame(index = window_df.index, columns =  window_df.columns)

    for col_name in residual_windows_df.columns:
        for i in residual_windows_df.index:
            event_key = pattern_matching_df[col_name][i]
            residual_windows_df[col_name][i] = window_df[col_name][i] - event_set[event_key]

    return residual_windows_df

def eg_extract_threshold_windows_df(df, window_df, window_length, window_stride, init_level = 0.8):
    threshold_df = pd.DataFrame(index = df.index, columns =  window_df.columns)

    for col_name in window_df.columns:
        if col_name in ('AC4', 'Refrigerator'):
            threshold_df[col_name] = [max(df[col_name])] * len(df[col_name])
        else:
            init = int(len(df)*0.9)
            threshold = spot_sy(df[col_name], num_init = init, init_level = init_level, risk = 1e-4)
            threshold_df[col_name] = threshold['t']

    threshold_windows_df = extract_windows_df(threshold_df, window_length, window_stride)
    
    return threshold_windows_df

def eg_residual_matching_df(residual_window_df, threshold_window_df):
    residual_matching_df = pd.DataFrame(index = residual_window_df.index, columns =  residual_window_df.columns)

    for col_name in residual_matching_df.columns:
        for i in residual_matching_df.index:
            condition = np.any((residual_window_df[col_name][i] - threshold_window_df[col_name][i]) > 0) 
            if condition == True:
                residual_matching_df[col_name][i] = 'Residual +'
            else:
                residual_matching_df[col_name][i] = 'Residual -'
    
    return residual_matching_df

def eg_event2graph_df(pattern_df, residual_df, evnet_set):
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
