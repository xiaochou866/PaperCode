import operator
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import Counter  # 用于获取一个向量中出现次数最多的元素
from itertools import permutations  # 用于生成一个ndarray中任意两个数字的排列
import time

from sklearn.preprocessing import MinMaxScaler

from util.ReductUtil import *
import warnings
warnings.filterwarnings("ignore")

# 文献: Attribute group for attribute reduction

def generateAttrGroup(X: np.ndarray) -> list[list[int]]:
    '''
    :param X: 数据集的条件属性部分
    :return: 条件属性C的分组情况
    '''
    _, conditionAttrNum = X.shape
    attrGroupNum = int(conditionAttrNum / 3)  # 这里设置分组的个数为条件属性个数/3
    attrGroup = [[] for i in range(attrGroupNum)]
    kmeansResult = KMeans(n_clusters=attrGroupNum, random_state=0).fit(X.T).labels_

    for idx, category in enumerate(kmeansResult):
        # idx指属性所在的索引 category表示属性聚类之后所在的组
        # 将聚类后属于同一个组的属性的索引放到一个组中
        attrGroup[category].append(idx)
    return attrGroup


def delEleWithAttrGroup(T_p, attGroup) -> set[int]:
    if len(T_p) == 0: return set()
    delEle = []
    for attr in T_p:
        for group in attGroup:
            if attr in group:
                delEle.extend(group)
    return set(delEle)


def reductionUseAttributeGroup(dataName: str, radius: float, index: str, stopCondition: str, X:np.ndarray, Y:np.ndarray):
    '''
    :param dataName: 用于标识是哪一个数据集
    :param radius: 将要进行邻域粗糙集的邻域半径
    :param index: 指标 POS(依赖度) CE(熵) NDI(Neighborhood Discrimination Index) NDER(Neighborhood Decision Error Rate)
    :param stopCondition: PRE(与上一次迭代的得分做对比) FULL(全属性对比)
    '''

    '''
        准备工作
    '''
    attrGroup = generateAttrGroup(X)  # 根据K-Means得到的属性分组
    decClasses = generateDecisionClasses(Y)
    _, conditionAttrNum = X.shape

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
        主体逻辑
    '''
    # 第一种暂停约束
    if stopCondition == "PRE":

        if scoreTrend == "UP":
            ops1 = operator.gt  # 大于
            ops2 = operator.le  # 小于等于
        elif scoreTrend == "DOWN":
            ops1 = operator.lt  # 小于
            ops2 = operator.ge  # 大于等于

        # region 本轮约简开始 做一些预备工作
        start_time = time.time()  # 程序开始时间
        # print("###############################################################################################"
        #             "\n开始本轮约简 对比算法1 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}".format(dataName, radius, index,
        #                                                                                     stopCondition))
        AT = set(range(conditionAttrNum))  # 全体属性集合
        A = set()  # 用于记录最终结果
        T = set(range(conditionAttrNum))  # T is the set of canditate attributes
        T_p = set()  # T' is used to record the potential reduct in one iteration

        preScore = -100 if scoreTrend == "UP" else 100  # 如果全局的评价指标是最大的则采用UP的方式 如果全局的评价指标是最小的则采用DOWN的方式

        cycleNum = 1
        # print("运行情况:")
        # endregion

        while True:
            # region 运行时间超过一定的界限 自动结束函数运行 进行下一次属性约简
            middle_time = time.time()
            run_time_long = (middle_time - start_time) / 60
            # print("本轮属性约简选择属性轮数:{} 已运行时间:{}分钟".format(cycleNum, run_time_long))
            cycleNum += 1
            if run_time_long > 120:
                print("到目前为止属性约简超过2小时 退出本次函数调用")
                return
            # endregion

            T = AT - A  # 除去已经选择的属性
            delEles = delEleWithAttrGroup(T_p, attrGroup) # 获得与已经选择的属性在同一个属性组的属性
            T = T - delEles  # 除去和T'中元素在一个组中的所有元素

            if len(T) == 0:  # 如果T为空跳到第三步对T'置空
                T = AT - A
                T_p = set()

            curScore = -100 if scoreTrend == "UP" else 100  # 记录从这次的备选属性集合中选出来属性加入集合后的得分
            selectedAttr = 0  # 记录本次选出来的属性
            for i, a in enumerate(T):
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | set([a])), index, ND)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            # print("现在属性集合得分为:", curScore)
            if ops2(curScore, preScore):
                break

            A.add(selectedAttr)
            T_p.add(selectedAttr)
            preScore = curScore

        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
        # 运行结果记录
        # print("\n运行结果:")
        # print("约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
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
        #         "\n开始本轮约简 算法1 参数为 数据集:{} 邻域半径:{} 指标:{} 中止条件:{}".format(dataName, radius, index, stopCondition))

        AT = set(range(conditionAttrNum))  # 全体属性集合
        A = set()  # 用于记录最终结果
        T = set(range(conditionAttrNum))  # T is the set of canditate attributes
        T_p = set()  # T撇 is used to record the potential reduct in one iteration

        fullAttrSetScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, None, index, ND)

        cycleNum = 1
        # print("运行情况:")
        # endregion

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

            T = AT - A  # 除去已经选择的属性
            delEles = delEleWithAttrGroup(T_p, attrGroup)
            T = T - delEles  # 除去和T'中元素在一个组中的所有元素

            if len(T) == 0:  # 如果T为空跳到第三步对T'置空
                T = AT - A
                T_p = set()

            curScore = 100 if scoreTrend == "DOWN" else -100
            selectedAttr = 0
            for i, a in enumerate(T):
                tmpScore = evaluteAttrSetScoreIntegration(decClasses, radius, X, Y, list(A | set([a])), index, ND)
                if ops1(tmpScore, curScore):
                    curScore = tmpScore
                    selectedAttr = a

            A.add(selectedAttr)
            T_p.add(selectedAttr)

            # print("现在属性集合得分为:", curScore)
            if ops2(curScore, fullAttrSetScore):
                break

        end_time = time.time()  # 程序结束时间
        run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
        # print("\n运行结果:")
        # print("本轮约简需要的时间为:{}秒, {}分钟".format(run_time_sec, run_time_sec / 60))
        # print("最终选出的属性约简为:{}".format(A))
        # print("最终选出的属性集在该指标下的得分为:{}".format(curScore))

    return A, preScore if stopCondition=="PRE" else curScore, run_time_sec



if __name__ == "__main__":
    # print("你好世界")
    # radiusArr = np.arange(0.03, 0.06, 0.03).tolist()
    # dataSets = ["iris","wine"]
    #
    # for dataPath in dataSets:
    #     # for index in ["POS", "CE", "NDI", "NDER"]:
    #     for index in ["POS"]:
    #         for stopCondition in ["PRE"]:
    #             reductionUseAttributeGroup(dataPath, radiusArr, index, stopCondition)

    # 使用某个数据集的单体 X y
    # 固定数据集 固定邻域半径 固定X 固定y
    # index: POS CE NDI NDER
    # stopCondition: PRE FULL

    path = '../DataSet_TEST/{}.csv'.format("wine")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    sampelNum, attrNum = data.shape

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    res = reductionUseAttributeGroup("wine", 0.2, "POS", "PRE", X, Y)
    print(res) # 返回结果顺序 约简 得分 时间

