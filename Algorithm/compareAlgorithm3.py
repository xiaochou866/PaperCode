import operator
import time

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler

from util.ReductUtil import *  # 求取属性约简常用的一些函数
from sklearn import preprocessing

# 文献: A novel approach to attribute reduction based on weighted
# Hu M, Tsang E C C, Guo Y, et al. A novel approach to attribute reduction based on weighted neighborhood rough sets[J]. Knowledge-Based Systems, 2021, 220: 106908.

def generateAttrWeightVector(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''
    :param A: 数据集的条件属性部分
    :param Y: 数据集的决策属性部分
    :return: 每个条件属性的重要程度 用于后续生成距离矩阵和邻域关系
    '''

    sampleName, C = A.shape  # C代表的是条件属性的数量

    ATxA = np.dot(A.T, A)
    ATxAValue = np.linalg.det(ATxA)

    if ATxAValue != 0:
        # A的转置*A可逆的情况
        V = np.linalg.solve(ATxA, np.dot(A.T, Y))
    else:
        # A的转置*A不可逆的情况
        V = np.linalg.solve(ATxA + np.eye(C), np.dot(A.T, Y))

    denominator = sum(abs(V))
    W = np.empty(C)  # 声明一个初始的numpy数组 注意需要给该数组中的每一个元素进行重新赋值
    for i in range(C):
        W[i] = (C * abs(V[i])) / denominator  # 式子(9)
    return W


# 两个向量之间的差值加上权重计算方式
def twoSampleWeightedDistance(u: np.ndarray, v: np.ndarray, W: np.ndarray):
    return (W * (u - v)) ** 2  # 求和再开根号就是11式


def reductionUseWeightedNeighborhood(dataName: str, radius: float, index: str, stopCondition: str, X:np.ndarray, Y:np.ndarray):
    '''
    :param dataName: 用于标识是哪一个数据集
    :param radius: 将要进行邻域粗糙集的邻域半径
    :param index: 指标 POS(依赖度) CE(熵) NDI(Neighborhood Discrimination Index) NDER(Neighborhood Decision Error Rate)
    :param stopCondition: PRE(与上一次迭代的得分做对比) FULL(全属性对比)
    '''

    '''
        准备工作
    '''

    W = generateAttrWeightVector(X, Y)  # 生成每个属性的权重向量

    decClasses = generateDecisionClasses(Y)
    _, C = X.shape  # C代表的是条件属性的数量

    # region 针对一个数据集只应该运行一次 针对第三个指标的特殊处理
    ND = set()  # 注意这里针对一个数据集只应该运行一次 针对第三个指标的特殊处理
    if index == "NDI":
        for decClass in decClasses:
            for p in permutations(decClass, 2):
                ND.add(p)
    # endregion

    indexToScoreTrend = {"POS": "UP", "CE": "DOWN", "NDI": "DOWN", "NDER": "DOWN"}  # 该字典用于标识各个指标是越大越好还是越小越好
    scoreTrend = indexToScoreTrend[index]

    '''
        主逻辑
    '''
    # 第一种暂停约束

    if stopCondition == "PRE":

        if scoreTrend == "UP":
            ops1 = operator.gt  # 大于
            ops2 = operator.le  # 小于等于
        elif scoreTrend == "DOWN":
            ops1 = operator.lt  # 小于
            ops2 = operator.ge  # 大于等于

        # region 本轮约简开始
        start_time = time.time()  # 程序开始时间
        # print("###############################################################################################"
        #         "\n开始本轮约简 算法3 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(dataName, radius, index,
        #                                                                                     stopCondition))
        red = set()
        AT = set(range(C))  # 全体属性集合

        cycleNum = 1
        # print("运行情况:")
        # endregion

        preScore = -100 if scoreTrend == "UP" else 100
        while True:
            middle_time = time.time()
            run_time_long = (middle_time - start_time) / 60
            # print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
            cycleNum += 1
            if run_time_long > 120:
                print("本轮属性约简超过2小时 退出本次函数调用")
                return

            candidate = AT - red

            curScore = -100 if scoreTrend == "UP" else 100
            selectedAttr = 0
            for i, a in enumerate(candidate):
                tmpCols = list(red | set([a]))
                tmpW = [W[e] for e in tmpCols]
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y,
                                                            tmpCols, index,
                                                            ND, tmpW)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            if ops2(curScore, preScore):
                break

            red = red | set([selectedAttr])
            preScore = curScore

        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

        # 运行结果记录
        # print("\n运行结果:")
        # print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        # print("最终选出的属性约简为:{}".format(red))
        # print("最终选出的属性集在该指标下的得分为:{}".format(preScore))
        # return red, preScore, run_time_sec

    # 第二种暂停约束
    elif stopCondition == "FULL":
        if scoreTrend == "UP":
            ops1 = operator.gt  # 大于
            ops2 = operator.ge  # 大于等于
        elif scoreTrend == "DOWN":
            ops1 = operator.lt  # 小于
            ops2 = operator.le  # 小于等于

        # region 本轮约简开始
        start_time = time.time()  # 程序开始时间
        # print("###############################################################################################"
        #         "\n开始本轮约简 对比算法3 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(dataName, radius, index,
        #                                                                                     stopCondition))
        red = set()
        AT = set(range(C))  # 全体属性集合
        cycleNum = 1
        # print("运行情况:")
        # endregion

        fullAttrSetScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, None, index, ND, W)
        while True:
            candidate = AT -red

            curScore = -100 if scoreTrend == "UP" else 100
            selectedAttr = 0

            for i, a in enumerate(candidate):
                tmpCols = list(red | set([a]))
                tmpW = [W[e] for e in tmpCols]
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y,
                                                            tmpCols, index,
                                                            ND, tmpW)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            red = red | set([selectedAttr])

            if ops2(curScore, fullAttrSetScore):
                break


        print("\n运行结果:")
        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
        # print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        # print("最终选出的属性约简为:{}".format(red))
        # print("最终选出的属性集在该指标下的得分为:{}".format(curScore))
        # return red, curScore, run_time_sec

    return red, preScore if stopCondition=="PRE" else curScore, run_time_sec


if __name__ == "__main__":
    # radiusArr = np.arange(0.03, 0.32, 0.03).tolist()
    # dataSets = ["CLL_SUB_111", "COIL20", "colon", "drivFace", "glass",
    #             "isolet1234", "leukemia", "lung", "ORL", "orlraws10P",
    #             "sonar", "TOX_171", "USPS", "warpAR10P", "wine"]
    #
    # dataSets = ["leukemia", "lung", "ORL", "orlraws10P",
    #             "sonar", "TOX_171", "USPS", "warpAR10P", "wine"]
    #
    # for dataPath in dataSets:
    #     for index in ["POS", "CE", "NDI", "NDER"]:
    #         for stopCondition in ["PRE", "FULL"]:
    #             reductionUseWeightedNeighborhood(dataPath, radiusArr, index, stopCondition)
    # print("你好世界")


    path = '../DataSet_TEST/{}.csv'.format("wine")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    sampelNum, attrNum = data.shape

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    res = reductionUseWeightedNeighborhood("wine", 0.2, "POS", "PRE", X, Y)
    print(res) # 返回结果顺序 约简 得分 时间
