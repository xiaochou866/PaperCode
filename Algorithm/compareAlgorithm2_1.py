import operator
import time

import numpy as np
from scipy.spatial.distance import squareform, pdist
from itertools import combinations  # 生成一个范围内的所有属性的两两组合用于生成相似度矩阵
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from util.ReductUtil import *  # 求取属性约简常用的一些函数

# 文献: Feature Selection Based on Neighborhood Discrimination Index (disSimilarity)

def getTwoAttrKnowledgeDistance(sampleNum: int, neighborRelation1: list[np.ndarray],
                                neighborRelation2: list[np.ndarray]) -> float:
    '''
    :param sampleNum: 数据集中的样本个数
    :param neighborRelation1: 由某一单一属性所生成的邻域关系
    :param neighborRelation2: 由某一单一属性所生成的邻域关系
    :return: 两个属性
    '''
    n = len(neighborRelation1)  # 获取样本数
    molecule = np.array([])  # P9 (3)分子部分
    for i in range(n):  # 遍历这在这两个单一属性下生成的每个样本的邻域 并作对称差 得到两个属性之间的相似程度
        molecule = np.concatenate((molecule, np.setxor1d(neighborRelation1[i], neighborRelation2[i])), axis=0)
    return len(molecule) / (sampleNum * sampleNum)


def generateDisSimilarityMatrix(X: np.ndarray, delta: float) -> np.ndarray:
    '''
    :param X: 数据集的条件属性部分
    :param delta: 指定的邻域半径
    :return: 衡量各个属性之间的不相似度矩阵
    '''

    sampleNum, conditionAttrNum = X.shape
    neighborRelationArr = []
    for i in range(conditionAttrNum):
        disMatrix = generateDisMatrix(X, [i])
        neighborRelation = generateNeighbor(disMatrix, delta)
        neighborRelationArr.append(neighborRelation)

    disSimilarity = np.zeros((conditionAttrNum, conditionAttrNum))
    twoAttrCombinations = list(combinations(range(conditionAttrNum), 2))  # 生成所有属性的组合

    for attrCombination in twoAttrCombinations:
        attr1 = attrCombination[0]
        attr2 = attrCombination[1]
        score = getTwoAttrKnowledgeDistance(sampleNum, neighborRelationArr[attr1], neighborRelationArr[attr2])
        disSimilarity[attr1][attr2] = score
        disSimilarity[attr2][attr1] = score
    return disSimilarity


def reductionUseDisSimilarity(dataName: str, radius: float, index: str, stopCondition: str, X:np.ndarray, Y:np.ndarray):
    '''
    :param path: 将要进行约简的数据的路径
    :param radiusArr:
    :param index: 指标 POS(依赖度) CE(熵) NDI(Neighborhood Discrimination Index) NDER(Neighborhood Decision Error Rate)
    :param stopCondition: PRE(与上一次迭代的得分做对比) FULL(全属性对比)
    '''

    '''
        准备工作
    '''

    decClasses = generateDecisionClasses(Y)
    sampleNum, conditionAttrNum = X.shape  # C代表的是条件属性的数量

    # region 针对一个数据集只应该运行一次 针对第三个指标的特殊处理
    ND = set()
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
        #         "\n开始本轮约简 算法2.1 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(dataName, radius, index,
        #                                                                                     stopCondition))

        AT = set(range(conditionAttrNum))  # 全体属性集合
        A = set()  # 用于记录最终结果

        disSimilarityMatrix = generateDisSimilarityMatrix(X, radius)  # 生成不相似度矩阵

        cycleNum = 1
        # print("运行情况:")
        # endregion

        preScore = -1000 if scoreTrend == "UP" else 1000
        while True:
            # region 运行时间超过一定的界限 自动结束函数运行 进行下一次属性约简
            middle_time = time.time()
            run_time_long = (middle_time - start_time) / 60
            # print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
            cycleNum += 1
            if run_time_long > 120:
                print("本轮属性约简超过2小时 退出本次函数调用")
                return
            # endregion

            candidateAttrSet = AT - A

            attrPairs = []  # 用来存储所有的B
            if len(candidateAttrSet) > 1:  # 如果备选属性个数大于一个
                for a in candidateAttrSet:  # For Each a 属于 AT-A line5
                    # 这里需要将a属性与AT-A中的其他属性结合
                    maxDisValWitha = -1
                    maxDisAttrWitha = 0
                    for b in candidateAttrSet:
                        if disSimilarityMatrix[a][b] > maxDisValWitha:
                            maxDisValWitha = disSimilarityMatrix[a][b]
                            maxDisAttrWitha = b
                    attrPairs.append(set([a, maxDisAttrWitha]))
            elif len(candidateAttrSet) == 1:  # 如果备选属性个数只有一个
                # attrPairs.append(set([candidateAttrSet[0]]))
                attrPairs.append(candidateAttrSet.copy())

            curScore = -1000 if scoreTrend == "UP" else 1000
            selectedPair = None
            for attrPair in attrPairs:
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | attrPair), index, ND)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedPair = attrPair

            if ops2(curScore, preScore):
                break
            preScore = curScore
            A = A | selectedPair if selectedPair != None else A

        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

        # print("\n运行结果:")
        # print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        # print("最终选出的属性约简为:{}".format(A))
        # print("最终选出的属性集在该指标下的得分为:{}".format(preScore))

        # return A, preScore, run_time_sec

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
            #         "\n开始本轮约简 算法2.1 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(path, radius, index,
            #                                                                                     stopCondition))
            # endregion

            AT = set(range(conditionAttrNum))  # 全体属性集合
            A = set()  # 用于记录最终结果

            disSimilarityMatrix = generateDisSimilarityMatrix(X, radius)  # 生成不相似度矩阵

            # region 运行循环开始
            cycleNum = 1
            # print("运行情况:")
            # endregion
            fullAttrSetScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, None, index, ND)
            while True:
                # region 运行时间超过一定的界限 自动结束函数运行 进行下一次属性约简
                # middle_time = time.time()
                # run_time_long = (middle_time - start_time) / 60
                # print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
                # cycleNum += 1
                # if run_time_long > 120:
                #     print("本轮属性约简超过2小时 退出本次函数调用")
                #     return
                # endregion

                candidateAttrSet = AT - A

                attrPairs = []  # 用来存储所有的B
                if len(candidateAttrSet) > 1:  # 如果备选属性个数大于一个
                    for a in candidateAttrSet:  # For Each a 属于 AT-A line5
                        # 这里需要将a属性与AT-A中的其他属性结合
                        maxDisValWitha = -1
                        maxDisAttrWitha = 0
                        for b in candidateAttrSet:
                            if disSimilarityMatrix[a][b] > maxDisValWitha:
                                maxDisValWitha = disSimilarityMatrix[a][b]
                                maxDisAttrWitha = b
                        attrPairs.append(set([a, maxDisAttrWitha]))  # 用于选择一个与属性a最不相似的属性构成组合
                elif len(candidateAttrSet) == 1:  # 如果备选属性个数只有一个
                    # attrPairs.append(set([candidateAttrSet[0]]))
                    attrPairs.append(candidateAttrSet.copy())

                # 评估每一个B加入到约简属性集中之后依赖度的变化
                curScore = -100 if scoreTrend == "UP" else 100
                selectedPair = None
                for attrPair in attrPairs:
                    tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | attrPair), index, ND)
                    if ops1(tmpScore, curScore):
                        curScore = tmpScore
                        selectedPair = attrPair

                A = A | selectedPair if selectedPair != None else A
                # 判断约束是否满足 如果满足返回月间 否则继续进行属性挑选 line 14
                if ops2(curScore, fullAttrSetScore):
                    break

        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒

        # print("\n运行结果:")
        # print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        # print("最终选出的属性约简为:{}".format(A))
        # print("最终选出的属性集在该指标下的得分为:{}".format(curScore))
        # return A, curScore, run_time_sec
    return A, preScore if stopCondition=="PRE" else curScore, run_time_sec



if __name__ == "__main__":
    # radiusArr = np.arange(0.03, 0.32, 0.03).tolist()
    # dataSets = ["CLL_SUB_111", "COIL20", "colon", "drivFace", "glass",
    #             "isolet1234", "leukemia", "lung", "ORL", "orlraws10P",
    #             "sonar", "TOX_171", "USPS", "warpAR10P", "wine"]
    #
    # for dataPath in dataSets:
    #     for index in ["POS", "CE", "NDI", "NDER"]:
    #         for stopCondition in ["PRE", "FULL"]:
    #             reductionUseDisSimilarity(dataPath, radiusArr, index, stopCondition)
    # print("你好世界")

    path = '../DataSet_TEST/{}.csv'.format("wine")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    sampelNum, attrNum = data.shape

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    res = reductionUseDisSimilarity("wine", 0.2, "POS", "PRE", X, Y)
    print(res) # 返回结果顺序 约简 得分 时间