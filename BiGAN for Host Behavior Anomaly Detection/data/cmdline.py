import logging
import os
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

#恶意进程索引
#error 66
error_line = [199, 263, 485, 569, 646, 650, 807, 887, 1395, 1607,
              1680,
              1936,
              2031,
              2074,
              2086,
              2213,
              2302,
              2528,
              2551,
              2762,
              2804,
              2877,
              3102,
              3129,
              3172,
              3298,
              3316,
              3406,
              3510,
              3794,
              3966,
              4068,
              4131,
              4241,
              4323,
              4373,
              4581,
              4856,
              4896,
              4973,
              4995,
              5065,
              5085,
              5127,
              5178,
              5191,
              5361,
              5405,
              5406,
              5768,
              5777,
              5804,
              5847,
              6073,
              6128,
              6154,
              6280,
              6286,
              6416,
              6463,
              6576,
              6595,
              6609,
              7045,
              7105,
              7249]

# Get training dataset for ali
def get_train(*args):
    x = np.load('data/cmdline_train.npy').astype(np.float32)
    temp = []
    for i in range(len(x)):
        temp.append(0)
    y = np.array(temp)
    x_train = x
    y_train = y
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    return x_train, y_train


# Get testing dataset for ali
def get_test(*args):
    # seed shuffling
    rng = np.random.RandomState(42)
    temp = []

    x = np.load('data/cmdline_test.npy').astype(np.float32)
    for i in range(len(x)):
        if i not in error_line:
            temp.append(0)
        else:
            temp.append(1)

    y = np.array(temp)
    x_major = x[y == 0]
    y_major = y[y == 0]
    x_minor = x[y == 1]
    y_minor = y[y == 1]

    # contaminate_rate : the empirical ratio of anomalous samples = anomalous / (normal + anomalous)
    contaminate_rate = len(y_minor) / (len(y_minor) + len(y_major))
    print(contaminate_rate)

    size_major = x_major.shape[0]
    inds = rng.permutation(size_major)
    x_major, y_major = x_major[inds], y_major[inds]

    size_minor = x_minor.shape[0]
    inds = rng.permutation(size_minor)
    x_minor, y_minor = x_minor[inds], y_minor[inds]

    x_test = np.concatenate((x_major, x_minor), axis=0)
    y_test = np.concatenate((y_major, y_minor), axis=0)

    size_test = x_test.shape[0]
    inds = rng.permutation(size_test)
    x_test, y_test = x_test[inds], y_test[inds]

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    return x_test, y_test, contaminate_rate
