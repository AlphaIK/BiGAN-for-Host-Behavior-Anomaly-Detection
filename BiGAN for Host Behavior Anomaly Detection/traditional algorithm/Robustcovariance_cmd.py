import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

c = 66/7262

# 恶意进程索引
# error 66
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

def get_train():
    x_train = np.load('../data/cmdline_train.npy').astype(np.float32)
    temp = []
    for i in range(len(x_train)):
        temp.append(1)
    y_train = np.array(temp)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    return x_train, y_train

def get_test():
    x_test = np.load('../data/cmdline_test.npy').astype(np.float32)
    temp = []
    for i in range(len(x_test)):
        if i in error_line:
            temp.append(-1)
        else:
            temp.append(1)
    y_test = np.array(temp)

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)
    return x_test, y_test

def tran(y):
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        else:
            y[i] = 1
    return y

start = time.time()
trainx, trainy = get_train()
testx, testy = get_test()
clf = EllipticEnvelope(contamination=c, random_state=np.random.RandomState(42))
clf.fit(testx)
y_pred = clf.predict(testx)

print(testy.shape, y_pred.shape)
print('testy -1:', (testy == -1).sum())
print('y_pred -1:', (y_pred == -1).sum())


testy = tran(testy)
y_pred = tran(y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(testy, y_pred, average='binary')
accuracy = accuracy_score(testy,y_pred)
print("Accuracy:", accuracy, 'Precision:', precision, 'Recall:', recall,'F1:', f1)
print("finish in {}s".format(time.time()-start))

