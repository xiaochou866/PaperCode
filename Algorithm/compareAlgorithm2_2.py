import operator

import numpy as np
from itertools import combinations
from sklearn import preprocessing
from util.ReductUtil import *  # 求取属性约简常用的一些函数
import time



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


def generateSimilarityMatrix(X: np.ndarray, delta: float) -> np.ndarray:
    '''
    :param X: 数据集的条件属性部分
    :param delta: 指定的邻域半径
    :return: 衡量各个属性之间的相似度矩阵
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

    return np.ones((conditionAttrNum, conditionAttrNum)) - disSimilarity  # 将单位矩阵减去不相似度矩阵得到相似度矩阵


def partitionAttrGroupBySimilarity(similarityMatrix: np.ndarray, k: int) -> list[set]:
    '''
    :param similarityMatrix: 表征属性间相似度的矩阵
    :param k: 将要划分的属性群的个数
    :return: 每一个属性群用集合进行表示 最终将这些属性群放到一个列表中
    '''
    # print(similarityMatrix)
    n = len(similarityMatrix)

    groupSize = round(n / k)  # 每个属性组中的属性个数  这里得保证groupSize>=2 否则该策略没有意义
    attrGroup = [set() for i in range(k)]

    visited = np.array([0 for i in range(n)])  # 标识属性是否已经选择
    for i in range(0, k - 1):
        tmp = set()

        # 先选出最相似的两个属性 放到属性组中
        waitSelectAttrs = np.where(visited == 0)[0].tolist()
        twoAttrCombinations = list(combinations(waitSelectAttrs, 2))  # 生成所有属性的组合
        # print(twoAttrCombinations)
        maxSim = -1
        maxCom = None
        for combination in twoAttrCombinations:
            x = combination[0]
            y = combination[1]
            if similarityMatrix[x][y] > maxSim:
                maxCom = [x, y]
                maxSim = similarityMatrix[x][y]

        tmp = tmp | set(maxCom)
        visited[maxCom[0]], visited[maxCom[1]] = 1, 1  # 设置这两个属性已经被访问过
        waitSelectAttrs.remove(maxCom[0])
        waitSelectAttrs.remove(maxCom[1])

        # print(groupSize)
        for j in range(2, groupSize):
            maxScore = -1
            maxAttr = None

            for attr in waitSelectAttrs:
                curScore = 0

                for e in tmp:
                    curScore += similarityMatrix[attr][e]

                if curScore > maxScore:
                    maxScore = curScore
                    maxAttr = attr

            # print(maxScore)
            # print(maxAttr)
            tmp = tmp | set([maxAttr])
            visited[maxAttr] = 1
            waitSelectAttrs.remove(maxAttr)
        attrGroup[i] = tmp
    attrGroup[k - 1] = set(np.where(visited == False)[0].tolist())

    return attrGroup


def reductionUseSimilarity(dataName: str, radius: float, index: str, stopCondition: str, X:np.ndarray, Y:np.ndarray):
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

    k = int(conditionAttrNum / 3) + 1  # 属性分组数


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
        print("###############################################################################################"
                "\n开始本轮约简 算法2.2 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(dataName, radius, index,
                                                                                            stopCondition))

        simMatrix = generateSimilarityMatrix(X, radius)
        attrGroup = partitionAttrGroupBySimilarity(simMatrix, k)

        A = set([])
        F = set([])
        AT = set(range(conditionAttrNum))

        # 增量约束模式
        preScore = -100 if scoreTrend == "UP" else 100

        cycleNum = 1
        print("运行情况:")
        # endregion

        while True:
            # region 运行时间超过一定的界限 自动结束函数运行 进行下一次属性约简
            middle_time = time.time()
            run_time_long = (middle_time - start_time) / 60
            print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
            cycleNum += 1
            if run_time_long > 120:
                print("本轮属性约简超过2小时 退出本次函数调用")
                return
            # endregion

            flag = False
            if len(F) == conditionAttrNum:
                candidate = AT - A
            else:
                candidate = AT - A - F
                flag = True

            curScore = -100 if scoreTrend == "UP" else 100
            selectedAttr = 0
            for i, a in enumerate(candidate):
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | set([a])), index, ND)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            if ops2(curScore, preScore):
                break
            preScore = curScore
            A = A | set([selectedAttr])

            if flag:
                for group in attrGroup:
                    if selectedAttr in group:
                        F = F | group

        # region 运行结果记录
        print("\n运行结果:")
        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
        print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        print("最终选出的属性约简为:{}".format(A))
        print("最终选出的属性集在该指标下的得分为:{}".format(preScore))

        return A, curScore, run_time_sec


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
        print("###############################################################################################"
                "\n开始本轮约简 算法2.2 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}\n".format(dataName, radius, index,
                                                                                            stopCondition))

        simMatrix = generateSimilarityMatrix(X, radius)
        attrGroup = partitionAttrGroupBySimilarity(simMatrix, k)

        A = set([])
        F = set([])
        AT = set(range(conditionAttrNum))

        cycleNum = 1
        print("运行情况:")
        # endregion
        fullAttrSetScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, None, index, ND)
        while True:

            # region 运行时间超过一定的界限 自动结束函数运行 进行下一次属性约简
            middle_time = time.time()
            run_time_long = (middle_time - start_time) / 60
            print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
            cycleNum += 1
            if run_time_long > 120:
                print("本轮属性约简超过2小时 退出本次函数调用")
                return
            # endregion

            flag = False
            if len(F) == conditionAttrNum:
                candidate = AT - A
            else:
                candidate = AT - A - F
                flag = True

            curScore = -100 if scoreTrend == "UP" else 100
            selectedAttr = 0
            for i, a in enumerate(candidate):
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | set([a])), index, ND)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            A = A | set([selectedAttr])

            if flag:
                for group in attrGroup:
                    if selectedAttr in group:
                        F = F | group
                        break

            if ops2(curScore, fullAttrSetScore):
                break
        # print(A)
        # print(curScore)
        # region 运行结果记录
        print("\n运行结果:")
        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
        print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        print("最终选出的属性约简为:{}".format(A))
        print("最终选出的属性集在该指标下的得分为:{}".format(curScore))
        return A, curScore, run_time_sec


if __name__ == "__main__":
    print("你好世界")
    # radiusArr = np.arange(0.03, 0.32, 0.03).tolist()
    # dataSets = ["CLL_SUB_111", "COIL20", "colon", "drivFace", "glass",
    #             "isolet1234", "leukemia", "lung", "ORL", "orlraws10P",
    #             "sonar", "TOX_171", "USPS", "warpAR10P", "wine"]
    #
    # for dataPath in dataSets:
    #     print(dataPath+"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     for index in ["POS", "CE", "NDI", "NDER"]:
    #         for stopCondition in ["PRE", "FULL"]:
    #             reductionUseSimilarity(dataPath, radiusArr, index, stopCondition)