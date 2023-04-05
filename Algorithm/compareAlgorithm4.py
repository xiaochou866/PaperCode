import math

import time
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler


# Variable radius neighborhood rough sets and attribure reduction

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


def generateVariableRadiusNeighbor(disMatrix: np.array, radiusArr: list[float]) -> dict:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    neighbors = dict()
    n, _ = disMatrix.shape
    for i in range(n):
        neighbors[i] = set(np.where(disMatrix[i, :] < radiusArr[i])[0]) - {i}
    return neighbors


def generateNewNeighbor(X:np.ndarray, decClasses, attrSet:list[int], delta:float, la:float)->dict:
    disMatrix = generateDisMatrixUnderAttrSet(X, attrSet)
    neighbors = generateNeighbor(disMatrix, delta)
    radiusArr = generateVariableRadius(neighbors, decClasses, delta, la)
    # print("各个样本的变精度半径为:{}".format(radiusArr))
    newNeighbors = generateVariableRadiusNeighbor(disMatrix, radiusArr)
    return newNeighbors


def generateDecisionClasses(Y:np.ndarray):
    '''
    :param Y: 数据集的决策属性部分
    :return: 各个决策类的集合
    '''
    decClasses = dict()
    decValues = np.unique(Y)
    for decValue in decValues:
        decClasses[decValue] = set(np.where(Y==decValue)[0])
    return decClasses

# 写一个函数 可以计算出数据集中每两个样本 在某一个属性子集上的距离
def generateDisMatrixUnderAttrSet(X: np.ndarray, attrSet: list[int]) -> np.array:
    '''
    :param X: 传入的原始数据集
    :param cols: 纳入计算距离的那些列
    :return: 对象之间的距离矩阵
    '''
    distMatrix = squareform(pdist(X[:, attrSet], metric='euclidean'))
    return distMatrix

def calNumerator(Dx: list[int]):
    numerator = 0
    for i in range(len(Dx)-1):
        for j in range(i+1, len(Dx)):
            numerator += abs(len(Dx[i]) - len(Dx[j]))
    return numerator


def generateVariableRadius(neigobors, decClasses, delta, la):
    radiusArr = [0] * len(neigobors)
    for i in range(len(neigobors)):
        neighborSampleNum = len(neigobors[i])
        Dx = [set()]*len(decClasses)
        px = 0
        for j in range(1,len(decClasses)+1):
            Dx[j-1] = neigobors[i] & decClasses[j]
            if len(Dx[j-1])!=0:
                px +=1
        numerator = calNumerator(Dx)
        if px ==1:
            radiusArr[i] = delta
        else:
            SB = numerator/(math.comb(px, 2)*neighborSampleNum+0.01)
            radiusArr[i] = delta*np.exp(-1*la*SB)
    return radiusArr

#   约简算法1
def getPos(decisionClasses: dict, neighbors: dict) -> list:
    '''
    :param decisionClasses: 决策类
    :param neighbors:  生成的邻域
    :return: D的B正域
    '''
    posBD = []
    for decisionClass in decisionClasses.values():
        for k in neighbors.keys():
            neighbor = neighbors[k]
            if neighbor.issubset(decisionClass):
                # print("对象{} 邻域{},被包含在决策类{}中\n".format(k, neighbor, decisionClass))
                posBD.append(k)
        # break
    # 去除重复项
    posBD = list(set(posBD))
    # print("D的B正域为{},其中元素个数为{}".format(posBD, len(posBD)))
    # print("条件属性集相对于决策属性集的重要度为:",len(posBD)/conditionData.shape[0])
    return posBD

def reductionUseVariableRadiusNeighborhoodRoughSet(X: np.ndarray, Y: np.ndarray, delta:float=0.2, la:float=0.5):
    # 从数据集中获取相关信息
    sampleNum, attrNum = X.shape
    decClasses = generateDecisionClasses(Y)

    A = set()
    AT = set(range(attrNum))
    preScore = -100

    start_time = time.time()  # 程序开始时间

    while True:
        curScore = -100
        selectedAttr = 0

        for a in AT - A:  #
            # print()
            neighbors = generateNewNeighbor(X, decClasses, list(A | set([a])), delta, la)
            tmpScore = len(getPos(decClasses, neighbors))

            if tmpScore > curScore:
                curScore = tmpScore
                selectedAttr = a
        # print(curScore, preScore)
        if curScore <= preScore:
            break
        preScore = curScore
        A = A | set([selectedAttr])
    end_time = time.time()  # 程序结束时间
    run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

    return A, preScore/sampleNum, run_time_sec


if __name__ == "__main__":
    # print("你好世界")
    path = '../DataSet_TEST/{}.csv'.format("wine")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    # print(Y==0.0)

    # 对Y进行处理
    # uniqueVal = np.unique(Y)
    # for i in range(len(uniqueVal)-1, -1, -1):
    #     Y[Y==uniqueVal[i]] = i+1
    # Y = Y.astype(int)

    red = reductionUseVariableRadiusNeighborhoodRoughSet(X, Y, 0.2)
    print(red)
