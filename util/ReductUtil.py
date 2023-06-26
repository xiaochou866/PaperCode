from collections import Counter
from itertools import permutations

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import os

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def isContain(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    '''
    :param arr1: 第一个索引数组 一般是指某个样本的邻域
    :param arr2: 第二个索引数组 一般是指某个决策类
    :return: arr1是否被arr2所包含

    demo:
        arr1 = np.array([1,2])
        arr2 = np.array([1,2,3])
        print(isContain(arr1, arr2))
    '''
    return np.all(np.in1d(arr1, arr2))


def generateDecisionClasses(Y: np.ndarray) -> list[np.ndarray]:
    '''
    :param Y: 数据集决策属性那一列
    :return:  U/D
    '''
    U_D = []
    lables = np.unique(Y)
    for lable in lables:
        U_D.append(np.where(Y == lable)[0])
    return U_D


def generateDisMatrix(X: np.ndarray, cols: list = None, W: np.ndarray = None) -> np.ndarray:
    '''
    :param W: 算法3中的权重矩阵
    :param conditonData: 数据条件属性部分
    :param cols: 纳入计算的那些列
    :return: 对象之间的距离矩阵(根据欧式距离)
    '''

    ret = None

    if W is None:  # 用于计算除算法3之外其他算法两个样本之间距离的计算 使用欧式距离
        ret = squareform(pdist(X[:, cols], metric='euclidean')) if cols != None \
            else squareform(pdist(X, metric='euclidean'))
    else:  # 用于计算算法3两个样本之间距离的计算 对应论文中的(11)
        ret = squareform(pdist(X[:, cols], lambda u, v: np.sqrt(((W * (u - v)) ** 2).sum()))) if cols != None \
            else squareform(pdist(X, lambda u, v: np.sqrt(((W * (u - v)) ** 2).sum())))

    return ret


def generateDisMatrixPartUWithAll(X: np.ndarray, partIdx: list[int], cols: list = None) -> np.ndarray:
    '''
    :param X: 数据集的条件属性部分
    :param partIdx: 部分样本的索引情况
    :param cols: 参与计算的所有属性
    :return: 计算给定partIdx与所有样本之间两两的距离矩阵
    '''

    if cols != None:
        X1 = X[partIdx]
        X1 = X1[:, cols]
        X2 = X[:, cols]
    else:
        X1 = X[partIdx]
        X2 = X

    ret = cdist(X1, X2, metric="euclidean")
    return ret


def generateNeighbor(disMatrix: np.ndarray, radius: float) -> list[np.ndarray]:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    n, _ = disMatrix.shape
    neighbors = [np.array([]) for i in range(n)]

    for i in range(n):
        neighbors[i] = np.where(disMatrix[i, :] < radius)[0]
        neighbors[i] = np.delete(neighbors[i], np.where(neighbors[i] == i))  # 该元素的邻域中删除该元素本身
    return neighbors


# ----------------------------------------------------------------------------------------------------------------------
def checkSampleInPos(neighbor: np.ndarray, decClasses: list[np.ndarray]) -> bool:
    '''
    :param neighbor: 某一个样本的邻域
    :param decClasses: 所有的决策类
    :return: 如果该样本的邻域完全落在一个决策类里面 说明该样本属于正域 返回True 否则返回false
    '''
    for decClass in decClasses:
        if isContain(neighbor, decClass):
            return True
    return False


def getPOS(neighbors: list[np.ndarray], decClasses: list[np.ndarray], *args) -> int:
    '''
    :param neighbors: 各个样本点的邻域关系
    :param decClasses: 决策类
    :return: 依赖度 指标1
    '''
    posDB = np.array([])
    for decClass in decClasses:
        for neighbor in neighbors:
            if isContain(neighbor, decClass):
                posDB = np.concatenate((posDB, neighbor), axis=0)

    posDB = np.unique(posDB)
    return len(posDB) / len(neighbors)


def getCE(neighbors: list[np.ndarray], decClasses: list[np.ndarray], *args) -> int:
    '''
    :param neighbors: 生成的邻域关系
    :param decClasses: 生成的所有决策类
    :return: 评估条件属性集的熵值大小 指标2
    '''

    def findXd(x: int, decClasses: list[np.ndarray]) -> np.ndarray:
        '''
        :param x: 样本
        :param decClasses: 所有的决策类
        :return: 找到该样本位于的决策类
        '''
        Xd = None  # [x]D
        for c in decClasses:
            if x in c:
                Xd = c
                break
        return Xd

    CE_BD = 0
    U = len(neighbors)
    for i in range(U):  # 遍历每一个元素 计算和式中的每一项
        Xd = findXd(i, decClasses)
        jiao = np.intersect1d(neighbors[i], Xd)  # 第二个得分的交集部分
        CE_BD += len(jiao) * np.log(len(jiao) / len(neighbors[i])) if len(neighbors[i]) != 0 and len(jiao) != 0 else 0
        # break
    return -CE_BD / U


def getNDI(neighbors: list[np.ndarray], decClasses: list[np.ndarray], ND: set[tuple]) -> int:
    '''
    :param ND:
    :param neighbors: 生成的邻域关系
    :param decClasses: 生成的所有决策类
    :return: Neighborhood/Conditional Discrimination Index 指标3
    '''

    U = len(neighbors)

    # ND = set()  # 注意这里针对一个数据集只应该运行一次
    # for decClass in decClasses:
    #     for p in permutations(decClass, 2):
    #         ND.add(p)

    NB = set()
    for i in range(U):
        for j in neighbors[i]:
            NB.add((i, j))

    return np.log(len(NB) / len(NB & ND)) if len(NB & ND) != 0 else 100


def getNDER(neighbors: list[np.ndarray], decClasses: list[np.ndarray], Y: np.ndarray) -> int:
    '''
    :param neighbors: 生成的邻域关系
    :param decClasses: 生成的所有决策类
    :param Y: 决策属性那一列
    :return: Neighborhood Decision Error Rate 指标4
    '''

    def predLabelByNeighbor(neighbor: np.ndarray, Y: np.ndarray) -> object:
        '''
        :param neighbor: 一个邻域 其中是样本的索引
        :param Y: 决策属性的取值
        :return: 邻域中的样本最多属于的类别
        '''
        preLabel = None
        if len(neighbor) == 0:  # 如果一个邻域中没有一个元素
            preLabel = np.random.choice(Y)
        else:
            preLabel = Counter(Y[neighbor]).most_common()[0][0]
        return preLabel

    NDER_BD = 0  # 根据邻域中的样本进行分类 打上的标签与真实标签不相符的样本的个数
    U = len(neighbors)
    for i in range(U):
        label = Y[i]
        predLabel = predLabelByNeighbor(neighbors[i], Y)
        NDER_BD += 1 if label != predLabel else 0
    return NDER_BD / U


# ----------------------------------------------------------------------------------------------------------------------
def evaluteAttrSetScoreIntegration(decClasses: list[np.ndarray], radius: float,
                                   X: np.ndarray, Y: np.ndarray,
                                   cols: list[int] = None,
                                   index: str = "POS",
                                   ND: set[tuple] = None,
                                   W: np.ndarray = None):
    indexToFun = {"POS": getPOS,
                  "CE": getCE,
                  "NDI": getNDI,
                  "NDER": getNDER}
    getScore = indexToFun[index]

    disMatrix = generateDisMatrix(X, cols, W)
    neighbors = generateNeighbor(disMatrix, radius)

    ret = None
    if index == "NDI":
        ret = getScore(neighbors, decClasses, ND)
    elif index == "NDER":
        ret = getScore(neighbors, decClasses, Y)
    else:
        ret = getScore(neighbors, decClasses)

    return ret


def classifierAssessReductPerformance(X: np.ndarray, Y: np.ndarray) -> dict:
    classifierNames = ["K近邻", "逻辑斯谛回归", "支持向量机", "朴素贝叶斯", "决策树"]
    classifiers = [KNeighborsClassifier(),
                   LogisticRegression(penalty='l2'),
                   svm.SVC(kernel='rbf', probability=True),
                   GaussianNB(),
                   tree.DecisionTreeClassifier()
                   ]

    classificationResults = dict()
    for name, classifier in zip(classifierNames, classifiers):
        classificationResults[name] = cross_val_score(classifier, X, Y, cv=10, scoring='f1_macro')
    # print(classificationResults)

    return classificationResults


def recordRes(algorithm: str, data: str, radius: float, index: str, stopCondition: str, red: set, score: float, runtime,
              classificationResults: dict = None):
    resFilePath = "../res/{}_{}_Res.csv".format(data, algorithm)
    exist = os.path.exists(resFilePath)

    resLine = ""
    classifierNames = ["K近邻", "逻辑斯谛回归", "支持向量机", "朴素贝叶斯", "决策树"]

    if not exist:  # 如果结果文件不存在 添加表头
        header = ["邻域半径", "指标", "终止条件", "约简结果", "最终得分",
                  "运行时间(s)", "运行时间(min)", "运行时间(hour)"]
        for classifierName in classifierNames:
            header.append(classifierName + "Accs")
            header.append(classifierName + "Avg")
        resLine += "|".join(header)
        resLine += "\n"

    resLine += "{}|{}|{}|{}|{}|".format(radius, index, stopCondition, red, score)
    resLine += "{}|{}|{}|".format(runtime, runtime / 60, runtime / 3600)
    if not classificationResults is None:
        for classifierName in classifierNames:
            clsRes = classificationResults[classifierName]
            resLine += "{}|{}|".format(list(clsRes), np.average(clsRes))
    resLine += "\n"

    resFile = open(resFilePath, 'a')  # 追加模式
    resFile.write(resLine)
    resFile.close()
