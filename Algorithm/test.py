import numpy as np
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from myAlgorithm3 import getMultiGranleClusterByKeamns, getGranules, getCurClusterResult

path = '../DataSet_TEST/{}.csv'.format("wine" )
data = np.loadtxt(path, delimiter=",", skiprows=1)

X = data[:, :-1]
X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
Y = data[:, -1]
Y = Y.astype(int)
sampleNum, attrNum = X.shape

clusterNums, granuleSampleNumThresholds = getGranules(X.shape[0])  # 生成多个粒度用于后序生成多个粒度下的聚类结果 [45, 23, 14, 11, 9] [4, 8, 13, 17, 22]
preMultiClusterStrucArr, preMultiClusterSampleNumArr = getMultiGranleClusterByKeamns(X, Y, [0], clusterNums, granuleSampleNumThresholds)
# print(preMultiClusterStrucArr)
# print(preMultiClusterSampleNumArr)


A =set([0])
for i in range(1, attrNum):
    print(list(A | set([i])))
    res = getCurClusterResult(X, Y, list(A | set([i])), preMultiClusterStrucArr)
    print(res)