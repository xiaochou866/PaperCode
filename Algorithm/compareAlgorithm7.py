from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from util.ReductUtil import *
import numpy as np


# 对应的论文: 王长忠_2019_近似推理_Attribute reduction based on k-nearest neighborhood rough sets
def calEachAttrStandardDeviation(X: np.ndarray) -> np.ndarray:
    '''
    对应Def3.1 用于计算两个样本之间的距离
    :param X: 数据集的条件属性部分
    :return: 各个属性的标准差
    '''
    attrNum = X.shape[1]
    stdDev = [0] * attrNum
    for i in range(attrNum):
        stdDev[i] = np.std(X[:, i])
    return stdDev


def generateDisMatrixPartUWithAll(X: np.ndarray, partIdx: list[int], cols: list = None,
                                    stdDev: np.ndarray = None) -> np.ndarray:
    '''
    :param X: 数据集的条件属性部分
    :param partIdx: 部分样本的索引情况
    :param cols: 参与计算的所有属性
    :param stdDev: 属性权重值
    :return: 计算给定partIdx与所有样本之间两两的距离矩阵
    '''

    if cols != None:
        X1 = X[partIdx]
        X1 = X1[:, cols]
        X2 = X[:, cols]
    else:
        X1 = X[partIdx]
        X2 = X

    if stdDev != None:  # 只有在传入各个属性的标准差的时候才会给各个属性给予权重
        W = [0] * len(cols)
        for i in range(len(cols)):
            W[i] = 1 / stdDev[cols[i]]
        ret = cdist(X1, X2, lambda u, v: np.sqrt(((W * (u - v)) ** 2).sum()))
    else:
        ret = cdist(X1, X2, metric="euclidean")
    return ret


def generateNeighborByDelta(disMatrix: np.ndarray, radius: float, partIdx: list[int]) -> list[np.ndarray]:
    '''
    :param disMatrix: 部分样本 与 全部样本 两两之间的距离矩阵
    :param radius: 用于生成邻域的半径 默认为1
    :param partIdx: 表示部分样本的索引
    :return:
    '''
    n = len(partIdx)
    neighbors = [np.array([]) for i in range(n)]

    for i in range(n):
        neighbors[i] = np.where(disMatrix[i, :] < radius)[0]
        neighbors[i] = np.delete(neighbors[i], np.where(neighbors[i] == i))  # 该元素的邻域中删除该元素本身
    return neighbors


def generateNeighborByKnearest(disMatrix: np.ndarray, k: int, partIdx: list[int]) -> list[np.ndarray]:
    '''
    :param disMatrix: 距离矩阵
    :param k: 用于指定是选择离该样本最近的多少个样本
    :return: 离各个样本最近的k个样本
    '''
    n = len(partIdx)
    neighbors = [np.array([]) for i in range(n)]

    for i in range(n):
        # https://blog.csdn.net/weixin_39604557/article/details/110832409
        neighbors[i] = np.argpartition(disMatrix[i], k + 1)[:k + 1]
        neighbors[i] = np.delete(neighbors[i], np.where(neighbors[i] == partIdx[i]))  # 该元素的邻域中删除该元素本身
    return neighbors


def generateNeighborByDeltaAndKnearest(X: np.ndarray, partIdx: list[int], cols: list = None,
                                        k: int = 3, stdDev: np.ndarray = None, radius: float = 1):
    '''
    :param X: 样本的条件属性部分
    :param partIdx: 计算邻域关系的那一部分样本的索引
    :param cols: 参与计算的属性
    :param k: 离样本最近的k个样本
    :param stdDev: 标准差
    :param radius: 半径用于生成邻域
    :return:
    '''

    mat1 = generateDisMatrixPartUWithAll(X, partIdx, cols, stdDev)
    neighbors1 = generateNeighborByDelta(mat1, radius, partIdx)
    mat2 = generateDisMatrixPartUWithAll(X, partIdx, cols)
    neighbors2 = generateNeighborByKnearest(mat2, k, partIdx)

    for i in range(len(neighbors1)):
        if len(neighbors1[i]) > k:
            neighbors1[i] = np.intersect1d(neighbors1[i], neighbors2[i])
    return neighbors1


def reductionUseNearestNeighborhood(X: np.ndarray, Y: np.ndarray, k: int, thelta: float = 0.01, radius: float = 1):
    sampelNum, attrNum = X.shape
    stdDev = calEachAttrStandardDeviation(X)
    decClasses = generateDecisionClasses(Y)

    allAttr = set(range(attrNum))
    red = set()
    U = set(range(sampelNum))
    V = U  # 使得论域不断缩减

    while True:  # 每次进入循环尝试选择一个最优价值的属性
        curScore = 0
        selectedAttr = 0
        curPos = set()

        candidateAttrSet = allAttr - red  # 备选属性集合
        for a in candidateAttrSet:
            pos = set()
            partIdx = sorted(list(V))
            neighbors = generateNeighborByDeltaAndKnearest(X, partIdx, list(red | set([a])), k,  stdDev)
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
    return red


if __name__ == "__main__":
    path = '../DataSet_TEST/ori/{}.csv'.format("vote")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    sampelNum, attrNum = X.shape
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    red = reductionUseNearestNeighborhood(X, Y, 50)
    print(red)
