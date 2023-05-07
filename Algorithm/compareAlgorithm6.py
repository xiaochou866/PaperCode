# Data-guided multi-granularity selector for attribute reduction

import math
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import squareform, pdist

# 现在要做的事情是在给定一个半径下得到全属性集合下各个样本的邻域
# 参数情况: 固定的delta = 0.2, w = 0.1, s = 10
# 写一个函数 可以计算出数据集中每两个样本 在某一个属性子集上的距离
def generateDisMatrixUnderAttrSet(X: np.ndarray, attrSet: list[int]) -> np.array:
    '''
    :param X: 传入的原始数据集
    :param cols: 纳入计算距离的那些列
    :return: 对象之间的距离矩阵
    '''
    distMatrix = squareform(pdist(X[:, attrSet], metric='euclidean'))
    return distMatrix

def generateNeighbor(X:np.ndarray, attrSet:list[int], radius: float) -> dict:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    disMatrix = generateDisMatrixUnderAttrSet(X, attrSet) # 生成该属性集之下的各个样本之间的距离矩阵
    neighbors = dict()
    n, _ = disMatrix.shape
    for i in range(n):
        # neighbors[i] = set(np.where(disMatrix[i, :] < radius)[0])
        neighbors[i] = set(np.where(disMatrix[i, :] < radius)[0]) - {i}
    return neighbors

def calDiffBetTwoGran(neighbors1:dict, neighbors2:dict)->float:
    sampleNum = len(neighbors1)
    diffVal = 0
    for i in range(sampleNum):
        s1 = neighbors1[i]
        s2 = neighbors2[i]
        symDif = s1.symmetric_difference(s2)
        diffVal += len(symDif)/sampleNum
    return diffVal/sampleNum


def dataGuidedParameterSelector(X:np.ndarray, s:int=10, w:int=0.01):
    '''
    :param X: 数据集的条件属性部分
    :param s: 默认要选出10个半径
    :param w: 差异值设置为0.1
    :return: 返回选出来具有差异的一些半径
    '''
    selectedRadius = []
    prePartRadius = [0.2-0.02*i for i in range(1, 10)]
    postPartRadius = [0.2+0.02*i for i in range(1, 11)]
    sampleNum, attrNum = X.shape
    AT = list(range(attrNum))
    neighborsWithDelta = generateNeighbor(X, AT, 0.2)

    preDiffAndRadArr = []
    for rad in prePartRadius:
        neighborsCompare = generateNeighbor(X, AT, rad)
        dif = calDiffBetTwoGran(neighborsCompare, neighborsWithDelta)
        preDiffAndRadArr.append([dif, rad])
    preDiffAndRadArr.sort(key=lambda pair: pair[0], reverse=True)
    preSelectedRadius = [e[1] for e in preDiffAndRadArr[2:7]]

    postDiffAndRadArr = []
    for rad in postPartRadius:
        neighborsCompare = generateNeighbor(X, AT, rad)
        dif = calDiffBetTwoGran(neighborsCompare, neighborsWithDelta)
        postDiffAndRadArr.append([dif, rad])
    postDiffAndRadArr.sort(key=lambda pair: pair[0], reverse=True)
    postSelectedRadius = [e[1] for e in postDiffAndRadArr[2:7]]
    selectedRadius = preSelectedRadius+postSelectedRadius

    return selectedRadius

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


def getAttrSetScoreDataGuidedSelector(X:np.ndarray, selectedRadius:list[float], decClasses:dict, attrSet:list[int]):
        score = 0
        for rad in selectedRadius:
            neighbors = generateNeighbor(X, attrSet, rad)
            tmpScore = len(getPos(decClasses, neighbors))
            score += tmpScore
        score = score / len(selectedRadius)
        return score


def reductionUseDataGuidedSelector(X:np.ndarray, Y:np.ndarray):
    sampleNum, attrNum = X.shape
    decClasses = generateDecisionClasses(Y)
    selectedRadius = dataGuidedParameterSelector(X)
    AT = set(range(attrNum))
    A = set() # 用于存储最后的约简结果
    preScore = -100

    fullAttrSetScore = getAttrSetScoreDataGuidedSelector(X, selectedRadius, decClasses, list(AT))
    # print("全属性集的得分为:{}".format(fullAttrSetScore))

    start_time = time.time()  # 程序开始时间
    while True:
        curScore = -100
        selectedAttr = 0
        for a in AT - A:
            tmpScore = getAttrSetScoreDataGuidedSelector(X, selectedRadius, decClasses, list(A | set([a])))
            if tmpScore > curScore:
                curScore = tmpScore
                selectedAttr = a
        A = A | set([selectedAttr])
        if curScore >= fullAttrSetScore:
            break

    # 删除属性的过程 如果从选出的属性集A中去除属性之后仍然满足约束就可以将一个属性从A中删除
    while True:
        nextA = A
        for a in A:
            tmpScore = getAttrSetScoreDataGuidedSelector(X, selectedRadius, decClasses, list(A - set([a])))
            if tmpScore >= fullAttrSetScore:
                nextA = nextA - set([a])
                break  # 每次只删除一个属性
        if len(A) == 1 or nextA == A:
            break
        A = nextA

    end_time = time.time()  # 程序结束时间
    run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

    return A, curScore/sampleNum, run_time_sec


if __name__ == "__main__":
    path = '../DataSet_TEST/ori/{}.csv'.format("movement")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    red = reductionUseDataGuidedSelector(X, Y)
    print(red)




    # sampleNum, attrNum = X.shape
    # selectedRadius = [0.12+0.02*i for i in range(10)]
    # decClasses = generateDecisionClasses(Y)
    #
    # AT = set(range(attrNum))
    # score1 = getAttrSetScoreDataGuidedSelector(X, selectedRadius, decClasses, list(AT))
    # print(score1)
    #
    # score2 = getAttrSetScoreDataGuidedSelector(X, selectedRadius, decClasses, list(AT-{3}))
    # print(score2)




