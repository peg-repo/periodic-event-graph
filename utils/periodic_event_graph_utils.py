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


def matrix_profile_set(data_dict, dict, top_k, periods):
    profiles_set = {}
    
    for col_name in data_dict.keys():
        key = '{} Profile'.format(col_name)
        window_size = periods * dict[col_name]
        profiles_set[key] = mp.compute(data_dict[col_name], window_size)
        profiles_set[key] = mp.discover.motifs(profiles_set[key],
                                               exclusion_zone=window_size // 2,
                                               k=top_k)
            
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
            distance_matrix[j, i] = distance_matrix[i, j]  # symmetric matrix
            
    return distance_matrix

def event_generation_set(data_dict, mp_set, dict, periods, cluster_size):
    # create a 2D array with padding
    motifs = {}
    for col_name in data_dict.keys():
        key = '{} Profile'.format(col_name)
        index = [item['motifs'] for item in mp_set[key]['motifs']]                
        motifs[col_name] = list(set(sum(index, [])))                    # exclude duplicate indices

    patterns = []
    for col_name in data_dict.keys():
        window_size = periods * dict[col_name]
        patterns.append([data_dict[col_name][idx : idx + window_size] for idx in motifs[col_name]])
    patterns = list({tuple(array) for array in sum(patterns, [])})      # exclude duplicate
    patterns = [np.array(array) for array in patterns]

    # create a 2D array with padding
    max_length = max(len(pattern) for pattern in patterns)
    padded_patterns = np.array([np.pad(pattern, (0, max_length - len(pattern)), mode='constant') for pattern in patterns])

    hdb_cluster = hdbscan.HDBSCAN(metric = dtw_distance, min_cluster_size = cluster_size)
    hdb_cluster.fit(padded_patterns)
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

def extract_windows_df(df, data_dict, dict, periods, stride, size):
    num_windows = (len(df) - size) // stride + 1
    range_windows = range(0, num_windows * stride, stride)
    windows_df = pd.DataFrame(index=range_windows, columns=df.columns)

    windows = []
    for col_name in data_dict.keys():
        windows = []
        window_size = periods * dict[col_name]
        for i in range_windows:
            data = data_dict[col_name][i: i + window_size]
            windows.append(data.tolist())
        windows_df[col_name] = windows

    return windows_df


def pattern_matching_df(window_df, event_set):
    similarity_tensor = {}
    
    for i in window_df.index:
        sliding_window = window_df.loc[i]
        similarity = {}
        
        for col_name in window_df.columns:
            ts_similarity = []
            for event in event_set.values():
                # Apply padding to adjust lengths when they differ
                if len(sliding_window[col_name]) < len(event):
                    padded_window = np.pad(sliding_window[col_name], (0, len(event) - len(sliding_window[col_name])), 'constant')
                    ts_similarity.append(mass2_sy(padded_window, event))
                elif len(sliding_window[col_name]) > len(event):
                    padded_event = np.pad(event, (0, len(sliding_window[col_name]) - len(event)), 'constant')
                    ts_similarity.append(mass2_sy(sliding_window[col_name], padded_event))
                else:
                    ts_similarity.append(mass2_sy(sliding_window[col_name], event))
                    
            similarity[col_name] = np.concatenate(ts_similarity)
            
        similarity_tensor[i] = similarity

    
    event_matching_df = pd.DataFrame(index=window_df.index, columns=window_df.columns)
    
    for i in similarity_tensor.keys():
        similarity = similarity_tensor[i]
        for col_name in similarity.keys():
            event_matching_df[col_name][i] = 'Event {}'.format(np.nanargmin(similarity[col_name]) + 1)

    return event_matching_df

def event2graph_df(pattern_df, residual_df, pattrn_event_set, residual_event_set):
    pattern_mapping = {event: idx for idx, event in enumerate(list(pattrn_event_set.keys()))}
    residual_mapping = {event: idx + len(pattrn_event_set.keys()) for idx, event in enumerate(list(residual_event_set.keys()))}
    timeseries_mapping = {time_series: idx for idx, time_series in enumerate(list(pattern_df.columns))}
    
    pattern_matching_mapped = pattern_df.copy() 
    for column in pattern_df.columns:
        pattern_matching_mapped[column] = pattern_df[column].map(pattern_mapping)
    
    residual_matching_mapped = residual_df.copy() 
    for column in residual_df.columns:
        residual_matching_mapped[column] = residual_df[column].map(residual_mapping)

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

def event2graph_not_df(pattern_df, pattrn_event_set):
    pattern_mapping = {event: idx for idx, event in enumerate(list(pattrn_event_set.keys()))}
    timeseries_mapping = {time_series: idx for idx, time_series in enumerate(list(pattern_df.columns))}
    
    pattern_matching_mapped = pattern_df.copy() 
    for column in pattern_df.columns:
        pattern_matching_mapped[column] = pattern_df[column].map(pattern_mapping)
    
    graph_df = pd.melt(pattern_matching_mapped.reset_index(), id_vars = ['index'], var_name = 'u', value_name = 'i').rename(columns={'index': 'ts'})
 
    graph_df = graph_df.reset_index()
    graph_df['label'] = 0.0  # state level
    graph_df = graph_df[['u', 'i', 'ts', 'label']]
    graph_df['u'] = graph_df['u'].map(timeseries_mapping)   # appiance mapping

    edge_features_df = pd.concat([pd.get_dummies(graph_df['u'], prefix='u', dtype=int),
                                  pd.get_dummies(graph_df['i'], prefix='i', dtype=int)], axis=1)
    graph_df = pd.concat([graph_df, edge_features_df], axis=1)
    graph_df = graph_df.sort_values(by=['ts', 'u'])
    graph_df = graph_df.reset_index(drop=True)

    return graph_df
    
def apply_stl_trend(df, periods):
    trend_dict = {}
    for col in df.columns:
        stl = STL(df[col], period=periods)
        result = stl.fit()
        trend_dict[col] = np.array(result.trend)
        
    return trend_dict
    
def apply_stl_seasonal(df, periods):
    seasonal_dict = {}
    for col in df.columns:
        stl = STL(df[col], period=periods)
        result = stl.fit()
        seasonal_dict[col] = np.array(result.seasonal)
        
    return seasonal_dict
    
def apply_stl_residual(df, periods):
    residual_dict = {}
    for col in df.columns:
        stl = STL(df[col], period=periods)
        result = stl.fit()
        residual_dict[col] = np.array(result.resid)
        
    return residual_dict

def fft_estimate_period(data):
    fft_result = fft(data)
    fft_freq = np.fft.fftfreq(len(data))

    dominant_freq_index = np.argmax(np.abs(fft_result)[:len(fft_freq)//2])
    dominant_freq = fft_freq[dominant_freq_index]

    if dominant_freq_index > 0:
        period_length = int(1 / abs(dominant_freq))  # treat as a constant
        return period_length
    else:
        return None 

def dominant_periods(dict):
    periods_dict = {}
    for col, data in dict.items():
        period_length = fft_estimate_period(data)
        periods_dict[col] = period_length
    return periods_dict
