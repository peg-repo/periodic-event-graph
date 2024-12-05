import numpy as np
from math import log

from pot import pot
from utils.grimshaw import grimshaw


def spot_sy(data:np.array, num_init:int, risk:float, init_level):
    ''' Streaming Peak over Threshold

    Reference: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        risk: detection level

    Returns:
        logs: 't' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'t': [], 'a': []}

    init_data = data[:num_init]
    rest_data = data[num_init:]

    z, t = pot(init_data, init_level) 
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['t'] = [z] * num_init

    for index, x in enumerate(rest_data):
        if x > z:
            logs['a'].append(index + num_init)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            if gamma == 0 or r <= 0 or sigma < 0:    # to avoid calculation errors
                z = t
            else:
                z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            k = k + 1

        logs['t'].append(z)
    
    return logs
