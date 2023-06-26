import math

import time
import numpy as np
from scipy.spatial.distance import squareform, pdist
from util.ReductUtil import *
from sklearn.preprocessing import MinMaxScaler


# Variable radius neighborhood rough sets and attribure reduction
# Zhang D, Zhu P. Variable radius neighborhood rough sets and attribute reduction[J]. International Journal of Approximate Reasoning, 2022, 150: 98-121.

def generateNeighbor(disMatrix: np.array, radius: float) -> dict:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    neighbors = dict()
    n, _ = disMatrix.shape
    for i in range(n):
        # neighbors[i] = set(np.where(disMatrix[i, :] < radius)[0])
        neighbors[i] = set(np.where(disMatrix[i, :] < radius)[0]) - {i}
    return neighbors


def generateVariableRadiusNeighbor(disMatrix: np.array, partIdx: list[int], radiusArr: list[float]) -> dict:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    n, _ = disMatrix.shape
    neighbors = [np.array([]) for i in range(n)]

    for i in range(n):
        neighbors[i] = np.where(disMatrix[i, :] < radiusArr[i])[0]
        neighbors[i] = np.delete(neighbors[i], np.where(neighbors[i] == partIdx[i]))
    return neighbors


def generateDecisionClasses(Y: np.ndarray):
    '''
    :param Y: 数据集的决策属性部分
    :return: 各个决策类的集合
    '''
    decClasses = dict()
    decValues = np.unique(Y)
    for decValue in decValues:
        decClasses[decValue] = set(np.where(Y == decValue)[0])
    return decClasses


def calNumerator(Dx: list[int]):
    '''
    :param Dx: 某一个邻域中属于各个决策类的样本的情况
    :return: (7)式的分子部分
    '''
    numerator = 0
    for i in range(len(Dx) - 1):
        for j in range(i + 1, len(Dx)):
            numerator += abs(len(Dx[i]) - len(Dx[j]))
    return numerator


def generateVariableRadius(neigobors, decClasses, delta, la):
    '''
    :param neigobors: 多个邻域关系
    :param decClasses: 所有的决策类
    :param delta: 原有的半径
    :param la: 参数lambda 对应公式(8)
    :return:
    '''
    radiusArr = [0] * len(neigobors)
    for i in range(len(neigobors)):
        neighborSampleNum = len(neigobors[i])
        Dx = [set()] * len(decClasses)
        px = 0
        for j in range(1, len(decClasses) + 1):
            Dx[j - 1] = neigobors[i] & decClasses[j] if j in decClasses else set()
            if len(Dx[j - 1]) != 0:
                px += 1
        numerator = calNumerator(Dx)
        if px == 1:
            radiusArr[i] = delta
        else:
            SB = numerator / (math.comb(px, 2) * neighborSampleNum + 0.01)
            radiusArr[i] = delta * np.exp(-1 * la * SB)
    return radiusArr


def generateNewNeighbor(X: np.ndarray, decClasses, partIdx: list[int], attrSet: list[int], delta: float,
                        la: float) -> dict:
    '''
    :param X: 样本条件属性
    :param decClasses: 决策类别
    :param partIdx: 考虑计算与所有样本之间距离的样本索引
    :param attrSet: 待评估的属性集
    :param delta: 0.2
    :param la: 0.5
    :return:
    '''
    disMatrix = generateDisMatrixPartUWithAll(X, partIdx, attrSet)
    neighbors = generateNeighbor(disMatrix, delta)
    radiusArr = generateVariableRadius(neighbors, decClasses, delta, la)
    newNeighbors = generateVariableRadiusNeighbor(disMatrix, partIdx, radiusArr)
    return newNeighbors


def checkSampleInPos(neighbor: np.ndarray, decClasses: list[np.ndarray]) -> bool:
    '''
    :param neighbor: 某一个样本的邻域
    :param decClasses: 所有的决策类
    :return: 如果该样本的邻域完全落在一个决策类里面 说明该样本属于正域 返回True 否则返回false
    '''
    for decClass in decClasses.values():
        if isContain(neighbor, np.array(list(decClass))):
            return True
    return False


def reductionUseVariableRadiusNeighborhoodRoughSet2(X: np.ndarray, Y: np.ndarray, delta: float = 0.12, la: float = 0.5,
                                                    thelta: float = 0.01):
    '''
    :param X:
    :param Y:
    :param delta: 原始半径
    :param la: 固定值用于生成变化后的半径
    :param thelta: 用于终止条件
    :return:
    '''
    sampelNum, attrNum = X.shape
    decClasses = generateDecisionClasses(Y)

    allAttr = set(range(attrNum))
    red = set()
    U = set(range(sampelNum))
    V = U  # 使得论域不断缩减

    start_time = time.time()  # 程序开始时间
    while True:  # 每次进入循环尝试选择一个最优价值的属性
        curScore = 0
        selectedAttr = 0
        curPos = set()

        candidateAttrSet = allAttr - red  # 备选属性集合
        for a in candidateAttrSet:
            pos = set()
            partIdx = sorted(list(V))  # 由于剩余论域中的样本使用集合进行存储的所以是无序的所以将其转化为列表并排序
            neighbors = generateNewNeighbor(X, decClasses, partIdx, list(red | set([a])), delta, la)
            for i in range(len(partIdx)):
                if checkSampleInPos(neighbors[i], decClasses):
                    pos.add(partIdx[i])

            tmpScore = len(pos)
            if tmpScore > curScore:
                curScore = tmpScore
                selectedAttr = a
                curPos = pos

        if curScore / sampelNum <= thelta:  # 如果当前轮次加入到论域U中的样本 与 U中样本数量的比值<=thelta则直接跳出
            break

        red.add(selectedAttr)
        V = V - curPos

    if len(red)==0:
        red.add(0)
    end_time = time.time()  # 程序结束时间
    run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

    return red, 0, run_time_sec


if __name__ == "__main__":
    # print("你好世界")
    path = '../DataSet_TEST/ori/{}.csv'.format("plrx")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    red = reductionUseVariableRadiusNeighborhoodRoughSet2(X, Y, 0.38)
    print(red)
