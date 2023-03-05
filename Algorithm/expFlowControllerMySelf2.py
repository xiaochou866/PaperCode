import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from Algorithm.myAlgorithm import multiReductByKMeans


# https://huaweicloud.csdn.net/637f7c9edacf622b8df85fef.html?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-1-116241388-blog-120377395.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-1-116241388-blog-120377395.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2
# 用于生成五折交叉验证的x和y
# dataNames = ["wine", "iris"]
dataNames = ["plrx", "sonar"]

for dataName in dataNames:
    f = open(r"..\Res\multiReductionByKmeans\{}.txt".format(dataName), "a", encoding="UTF-8")
    f.write(dataName + '\n')
    path = '../DataSet_TEST/{}.csv'.format(dataName)
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    rowNum, colNum = data.shape
    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    y = data[:, -1]
    # print(X, y)

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(list(range(rowNum))):  # 调用split方法切分数据
        foldX = X[train_index, :]
        foldy = y[train_index]
        res = multiReductByKMeans(foldX, foldy)
        f.write(str(res) + '\n')
    f.close()
