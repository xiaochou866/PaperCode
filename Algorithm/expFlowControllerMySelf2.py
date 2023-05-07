import os
import datetime

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from Algorithm.myAlgorithm import multiReductByKMeans
# from Algorithm.myAlgorithm5 import acceAttrRedByMultiGranleKMeans
from Algorithm.myAlgorithm6 import acceAttrRedByMultiGranleKMeans
from sklearn.model_selection import cross_val_score
from sklearn import tree,svm


def getXY(path: str):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    Y = Y.astype(int)
    return X, Y

def getClassificationAccuracy(oriX, oriY, attrSet) -> list[float]:
    knn = KNeighborsClassifier(n_neighbors=10)
    clf = svm.SVC(kernel='linear', C=1)
    cart = tree.DecisionTreeClassifier(max_depth=5)
    score1 = cross_val_score(knn, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    score2 = cross_val_score(clf, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    score3 = cross_val_score(cart, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    return score1, score2, score3

if __name__ == "__main__":
    dir = "..\DataSet_TEST"
    dataCategory = ["ori", "1fold", "2fold", "3fold", "4fold", "5fold", "10noise", "20noise", "30noise"]
    dataSet = ['fertility_Diagnosis', 'BreastTissue', 'Iris', 'wine', 'plrx',
                'GlaucomaM', 'Sonar', 'seeds', 'Glass', 'accent',
                'PimaIndiansDiabetes', 'Ionosphere', 'movement', 'vote', 'musk',
                'wdbc', 'diamonds_filter', 'australian', 'BreastCancer', 'diabetes',
                'pima', 'College', 'Vehicle', 'german', 'data_banknote', 'waveform']

    category = dataCategory[0]  # 指定数据的类别 每次专注于一种类别的数据集
    algorithmName = "myAlgorithm6"
    proportions = [0.025, 0.05, 0.075, 0.1, 0.125]
    for data in dataSet:
        # 原始数据
        oriPath = os.path.join(dir, "ori", "{}.csv".format(data))
        oriX, oriY = getXY(oriPath)

        # 指定的某一类别的数据
        path = os.path.join(dir, category, "{}.csv".format(data))
        X, Y = getXY(path)
        sampleNum, attrNum = X.shape

        nowTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        red, finScore, runTime = acceAttrRedByMultiGranleKMeans(X, Y, proportions)

        knnScore, svmScore, cartScore = getClassificationAccuracy(oriX, oriY, red)

        resList = [nowTime, algorithmName, category, data, proportions,
                    red, attrNum, len(red), finScore, runTime,
                    knnScore, svmScore, cartScore, sum(knnScore)/len(knnScore), sum(svmScore)/len(svmScore), sum(cartScore)/len(cartScore)]
        resList = [str(e) for e in resList]

        print(resList)  # 展示运行进度

        resStorePath = os.path.join("..\Res", "myAlgorithms", algorithmName, category, )
        if not os.path.exists(resStorePath):  # 检查目录是否存在如果不存在则创建目录
            os.makedirs(resStorePath)

        fileName = os.path.join(resStorePath, "{}.csv".format(data))
        if not os.path.exists(fileName):
            with open(fileName, "w", encoding="utf-8") as f:
                f.write("结果生成时间|算法名称|数据集种类|数据集名称|粒度生成比例|约简结果|原始属性长度|约简属性长度|属性集得分|运行时间|5折KNN准确率|5折SVM准确率|5折CART准确率|KNN平均准确率|SVM平均准确率|CART平均准确率\n")
                f.write("|".join(resList) + "\n")
        else:
            with open(fileName, "a", encoding="utf-8") as f:  # 默认会进行追加
                f.write("|".join(resList) + "\n")
