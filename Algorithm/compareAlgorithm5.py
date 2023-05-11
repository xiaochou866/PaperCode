# Random Sampling for Attribure Reduction(RSAR)


import math
import time
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

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

def reductionUseNeighborhoodRoughSetUnderPartU(X: np.ndarray, Y: np.ndarray, A:set, radius:float=0.2, compareScore:float=0, stopCondition:str="PRE"):
    # 从数据集中获取相关信息
    sampleNum, attrNum = X.shape
    decClasses = generateDecisionClasses(Y)
    AT = set(range(attrNum))

    if stopCondition == "PRE": # 添加属性之后得到的正域比上一次正域中的样本数量多就认为该属性是有效的加入其中
        preScore = compareScore
        while True:
            curScore = -100
            selectedAttr = 0
            candidateAttrSet = AT - A
            for a in candidateAttrSet:  #
                # print()
                neighbors = generateNeighbor(X, list(A | set([a])), radius)
                tmpScore = len(getPos(decClasses, neighbors))

                if tmpScore > curScore:
                    curScore = tmpScore
                    selectedAttr = a
            if curScore <= preScore:
                break
            preScore = curScore
            A = A | set([selectedAttr])

    elif stopCondition == "FULL":
        neighbors = generateNeighbor(X, list(AT), radius)
        fullAttrSetScore = len(getPos(decClasses, neighbors))
        while True: # 添加属性的过程
            candidateAttrSet = AT - A
            curScore =  -100
            selectedAttr = 0
            for a in candidateAttrSet:
                neighbors = generateNeighbor(X, list(A|set([a])), radius)
                tmpScore = len(getPos(decClasses, neighbors))
                if tmpScore>=curScore:
                    curScore = tmpScore
                    selectedAttr = a

            # print("现在属性集合得分为:", curScore)
            A.add(selectedAttr)

            if curScore>=fullAttrSetScore:
                break
    return A, preScore if stopCondition=="PRE" else fullAttrSetScore

# 用于生成五折交叉验证的x的索引
def generateFoldXindex(sampleNum:int, foldNum:int):
    '''
    :param sampleNum: 该数据集的样本数量
    :param foldNum: 进行数据集划分的折数
    :return: 各个不同论域
    '''
    testIndexArr = []
    kf = KFold(n_splits=foldNum, shuffle=False)
    for _, test_index in kf.split(list(range(sampleNum))):  # 调用split方法切分数据
        testIndexArr.append(test_index)
    return testIndexArr

# 基于经典的邻域粗糙集模型进行加速
def reductionUseRandomSampling(X:np.ndarray, Y:np.ndarray, radius:float=0.1, stopCondition:str="PRE"):
    decClasses = generateDecisionClasses(Y)
    foldNum = 5
    sampelNum = X.shape[0]
    testIndexArr = generateFoldXindex(sampelNum, foldNum)
    sampleIndexs = np.array([])
    A = set()

    start_time = time.time()  # 程序开始时间
    # compareScore = 0
    compareScore = -100
    for i in range(foldNum): # 不断将划分的各个部分加入到要考虑的论域中
        # print(compareScore)
        # print("本次循环的属性集合为{}".format(A))
        sampleIndexs = np.append(sampleIndexs, testIndexArr[i]).astype(int)
        # print(sampleIndexs)
        A, compareScore = reductionUseNeighborhoodRoughSetUnderPartU(X[sampleIndexs,:], Y, A, radius, compareScore)

    # 删除属性的过程 如果从选出的属性集A中去除属性之后仍然满足约束就可以将一个属性从A中删除
    while True:
        nextA = A
        for a in A:
            neighbors = generateNeighbor(X, list(A - set([a])), radius)
            tmpScore = len(getPos(decClasses, neighbors))
            if tmpScore >= compareScore:
                nextA = nextA - set([a])
                compareScore = tmpScore
                break  # 每次只删除一个属性
        if len(A) == 1 or nextA == A:
            break
        A = nextA
    end_time = time.time()  # 程序结束时间
    run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

    return A, compareScore/sampelNum, run_time_sec

if __name__ == "__main__":
    # print("你好世界")
    path = '../DataSet_TEST/10noise/{}.csv'.format("accent")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    # 对Y进行处理
    red = reductionUseRandomSampling(X, Y, 0.38, "PRE")
    print(red)


