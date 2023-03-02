import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


#   通用函数
def readSplitData(path: str, dCol=-1) -> pd.DataFrame:
    '''
    :param path: 要读取的数据文件的路径
    :param dCol: 决策属性所在的列 默认为最后一列
    :return: 要读取数据 的 条件属性部分 和 决策属性部分
    '''
    df = pd.read_csv(path, header=None)
    rowNum, colNum = df.shape

    decisionCol = [dCol]
    conditionCol = list(range(colNum))
    del conditionCol[dCol]
    return df.iloc[:, conditionCol], df.iloc[:, decisionCol]


def dataProcess(df: pd.DataFrame) -> np.array:
    '''
    :param df: 要处理的条件属性部分
    :return: 经过缺失值处理和归一化之后的条件属性部分
    '''
    # 填充缺失值
    # df.fillna(df.mean())
    # 归一化
    zscore = preprocessing.MinMaxScaler()
    zscore = zscore.fit_transform(df)
    df_zscore = pd.DataFrame(zscore, index=df.index, columns=df.columns)
    return np.array(df_zscore)


def generateDecisionClasses(decisionData: pd.DataFrame) -> dict:
    '''
    :param decisionData: 要处理数据的决策属性部分
    :return: 字典 键 为决策属性某一取值 值为 对应该决策属性取值的对象的集合
    '''
    Classes = dict()
    decisionData.columns = ['decision']
    for v in decisionData.decision.unique():
        Classes[v] = set(decisionData[decisionData.decision == v].index.values)
    return Classes


def generateDisMatrix(conditonData: np.array, cols: list) -> np.array:
    '''
    :param conditonData: 数据条件属性部分
    :param cols: 纳入计算的那些列
    :return: 对象之间的距离矩阵
    '''
    conditonData = conditonData[:, cols]
    tmpDist = pdist(conditonData, metric='euclidean')
    distMatrix = squareform(tmpDist)
    return distMatrix


def generateNeighbor(disMatrix: np.array, radius: float) -> dict:
    '''
    :param disMatrix: 距离矩阵
    :param radius:  用来生成邻域的半径
    :return:  字典 键:对象索引 值:该对象所生成邻域中的对象
    '''
    neighbors = dict()
    n, _ = disMatrix.shape
    for i in range(n):
        neighbors[i] = set(np.where(disMatrix[i, :] < radius)[0]) - {i}
    return neighbors


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


def FARNeM(decisionClasses: dict, conditionData: np.array, radius: float) -> list:
    '''
    胡 基于邻域的前向属性约简算法
    :param decisionClasses: 决策类
    :param conditionData: 条件属性部分数据
    :param radius: 邻域半径
    :return: 得到 指定邻域半径 对应的约简
    '''
    start = time.time()  # func开始的时间
    # 参与约简的属性越多 D的B正域先增大后减小
    red = []  # 约简 初始化为空
    attrs = list(range(conditionData.shape[1]))  # 所有属性

    preSIG = 0
    while len(attrs) > 0:
        n = len(attrs)
        importanceDegree = [0] * n
        for i in range(n):
            tmpRed = red.copy()
            tmpRed.append(attrs[i])
            # 生成指定列的距离矩阵 以及 每个对象的邻域
            disMatrix = generateDisMatrix(conditionData, tmpRed)
            neighbors = generateNeighbor(disMatrix, radius)
            posBD = getPos(decisionClasses, neighbors)
            importanceDegree[i] = len(posBD)
        # print("现在各个待添加的属性的重要程度为:", importanceDegree) # 展示添加属性的过程
        curSIG = max(importanceDegree)
        maxi = importanceDegree.index(curSIG)
        if curSIG > preSIG:
            red.append(attrs[maxi])
            # print("属性约简为:", red) # 展示添加属性的过程
            del attrs[maxi]
            preSIG = curSIG
        else:
            break
    end = time.time()  # func结束的时间
    print("数据集:{}\n约简算法1 邻域半径:{} 属性约简:{} 约简所用时间:{}ms".format(os.path.basename(DATA_PATH), radius, red,
                                                             (end - start) * 1000))
    return red


# 约简算法2
def getNeighborhoodEntropy(decisionClasses: dict, neighbors: dict) -> float:
    '''
    :param decisionClasses: 决策类
    :param neighbors: 生成的邻域
    :return: 邻域熵
    '''
    # 参与运算的属性越多 系统越不混乱 其值越小
    n = conditionData.shape[0]  # -1/n

    sum = 0
    for i in range(n):
        # print("xi的邻域为{}, 对应的决策类为{}".format(D_xi, NB_xi))
        decisionValue = decisionData.iloc[i][0]  # 该对象决策属性的取值
        D_xi = decisionClasses[decisionValue]  # 该对象所处的决策类
        NB_xi = neighbors[i]  # 该对象所生成的邻域
        if len(NB_xi) == 0 or len(D_xi & NB_xi) == 0:
            continue
        else:
            ratio = len(D_xi & NB_xi) / len(NB_xi)
            sum += np.log(ratio)
    return -sum / n


def reductionBaseNE(decisionClasses: dict, conditionData: np.array, radius: float) -> list:
    '''
    :param decisionClasses: 决策类
    :param conditionData: 条件属性部分数据
    :param radius: 邻域半径
    :return: 得到 指定邻域半径 对应的约简
    '''
    start = time.time()  # func开始的时间
    # 循环中止的条件
    deta = 0.01  # 约简算法2的参数

    red = []  # 初始条件下约简为空集
    attrs = list(range(conditionData.shape[1]))  # 所有属性

    preSIG = 100
    while len(attrs) > 0:
        n = len(attrs)
        importanceDegree = [0] * n
        for i in range(n):
            tmpRed = red.copy()
            tmpRed.append(attrs[i])
            # 生成指定列的距离矩阵 以及 每个对象的邻域
            disMatrix = generateDisMatrix(conditionData, tmpRed)
            neighbors = generateNeighbor(disMatrix, radius)
            importanceDegree[i] = getNeighborhoodEntropy(decisionClasses, neighbors)
        # print("现在各个待添加的属性的重要程度为:", importanceDegree) # 展示添加属性的过程
        curSIG = min(importanceDegree)
        mini = importanceDegree.index(curSIG)

        if preSIG - curSIG > deta:  # 邻域熵下降的幅度大于deta
            red.append(attrs[mini])
            # print("属性约简为:", red) # 展示添加属性的过程
            del attrs[mini]
            preSIG = curSIG
        else:
            break

    end = time.time()  # func结束的时间
    print("数据集:{}\n约简算法2 邻域半径:{} 属性约简:{} 约简所用时间:{}ms".format(os.path.basename(DATA_PATH), radius, red,
                                                             (end - start) * 1000))
    return red


# 整合
def Reduction1(decisionClasses: dict, conditionData: np.array, radius: int):
    FARNeM(decisionClasses, conditionData, radius)


def Reduction2(decisionClasses: dict, conditionData: np.array, radius: int):
    reductionBaseNE(decisionClasses, conditionData, radius)


def neigborhoodClassifier0(decisionClasses: dict, conditionData: np.array, radius: int):
    # 选取全部属性进行邻域分类
    print("数据集:{} 邻域半径:{}".format(os.path.basename(DATA_PATH), radius))

    start = time.time()  # func开始的时间
    cols = list(range(conditionData.shape[1]))
    disMatrix = generateDisMatrix(conditionData, cols)
    neighbors = generateNeighbor(disMatrix, radius)

    predict = []
    for k in neighbors.keys():
        similarPoints = list(neighbors[k])
        valueCounts = decisionData.iloc[similarPoints, :].value_counts()
        if (valueCounts.shape[0] == 0):
            if (type(decisionData.iloc[0][0]) == np.int64):
                predict.append(-1)
            else:
                predict.append("None")
        else:
            predict.append(valueCounts.idxmax()[0])
    end = time.time()  # func结束的时间

    original = [e[0] for e in decisionData.values.tolist()]
    print("原有对象类别:", original)
    print("预测对象类别:", predict)
    print("分类算法(全属性) 准确率:{}% 分类所用时间:{}ms".format(accuracy_score(original, predict) * 100, (end - start) * 1000))
    return predict


def neigborhoodClassifier1(decisionClasses: dict, conditionData: np.array, radius: float):
    # 先用约简算法1约简之后再进行邻域分类
    red = FARNeM(decisionClasses, conditionData, radius)

    start = time.time()  # func开始的时间
    disMatrix = generateDisMatrix(conditionData, red)
    neighbors = generateNeighbor(disMatrix, radius)

    predict = []
    for k in neighbors.keys():
        similarPoints = list(neighbors[k])
        valueCounts = decisionData.iloc[similarPoints, :].value_counts()
        if (valueCounts.shape[0] == 0):
            # print("节点{}的邻域为空".format(k))
            if (type(decisionData.iloc[0][0]) == np.int64):
                predict.append(-1)
            else:
                predict.append("None")
        else:
            predict.append(valueCounts.idxmax()[0])
    end = time.time()  # func结束的时间

    original = [e[0] for e in decisionData.values.tolist()]
    print("原有对象类别:", original)
    print("预测对象类别:", predict)
    print("分类算法1 准确率:{}% 分类所用时间:{}ms".format(accuracy_score(original, predict) * 100, (end - start) * 1000))
    return predict


def neigborhoodClassifier2(decisionClasses: dict, conditionData: np.array, radius: float):
    # 先用约简算法2约简之后再进行邻域分类
    red = reductionBaseNE(decisionClasses, conditionData, radius)

    start = time.time()  # func开始的时间
    disMatrix = generateDisMatrix(conditionData, red)
    neighbors = generateNeighbor(disMatrix, radius)

    predict = []
    for k in neighbors.keys():
        similarPoints = list(neighbors[k])
        valueCounts = decisionData.iloc[similarPoints, :].value_counts()
        if (valueCounts.shape[0] == 0):
            if (type(decisionData.iloc[0][0]) == np.int64):
                predict.append(-1)
            else:
                predict.append("None")
        else:
            predict.append(valueCounts.idxmax()[0])
    end = time.time()  # func结束的时间

    original = [e[0] for e in decisionData.values.tolist()]
    print("原有对象类别:", original)
    print("预测对象类别:", predict)
    print("分类算法2 准确率:{}% 分类所用时间:{}ms".format(accuracy_score(original, predict) * 100, (end - start) * 1000))
    return predict


# 测试选取一个较为合适的邻域半径
def selectProperRadius(start: int, end: int, step: float, testFun):
    radius = []

    cur = start
    while (cur <= end):
        radius.append(cur)
        cur += step
    print("进行测试的半径为:", radius)

    for r in radius:
        print("################################################################################################################")
        testFun(decisionClasses, conditionData, r)


if __name__ == "__main__":
    '''
        指定参数测试 函数 功能
    '''
    # 指定算法参数
    DATA_PATH = "./Sonar.csv"
    radius = 0.65

    # 数据处理
    conditionData, decisionData = readSplitData(DATA_PATH)  # 将数据分为条件属性部分 和 决策属性部分
    conditionData = dataProcess(conditionData)  # 对条件属性部分的数据进行缺失值填充和归一化 并将其转换为np.array类型变量 便于计算距离矩阵
    decisionClasses = generateDecisionClasses(decisionData)  # 使用决策属性部分的数据生成决策类

    # 指定相关算法
    # 约简
    # Reduction1(decisionClasses, conditionData, radius)
    # Reduction2(decisionClasses, conditionData, radius)

    # 全属性分类
    # neigborhoodClassifier0(decisionClasses, conditionData, radius)

    # 约简+分类
    # neigborhoodClassifier1(decisionClasses, conditionData, radius)
    neigborhoodClassifier2(decisionClasses, conditionData, radius)


    '''
        选择不同的邻域半径记录实验 
        4个数据集
        3个函数
        每个数据集 每个函数 邻域半径选取0-5 步长0.05
        4*3*100 12个txt 1200条测试
    '''
    # srcFiles = ["./example.csv", "./Sonar.csv", "./winequality-white.csv", "./magic04.csv", "./heart.csv"]
    # stdOut = sys.stdout  # 保留标准输出
    # for srcFile in srcFiles:
    #     DATA_PATH = srcFile
    #     conditionData, decisionData = readSplitData(DATA_PATH)  # 将数据分为条件属性部分 和 决策属性部分
    #     conditionData = dataProcess(conditionData)  # 对条件属性部分的数据进行缺失值填充和归一化 并将其转换为np.array类型变量 便于计算距离矩阵
    #     decisionClasses = generateDecisionClasses(decisionData)  # 使用决策属性部分的数据生成决策类
    #
    #     # function_names = ["neigborhoodClassifier0", "neigborhoodClassifier1", "neigborhoodClassifier2"]
    #     function_names = ["neigborhoodClassifier2"]
    #     for funName in function_names:
    #         # 重定向标准输出
    #         outFile = open("testResult_{}_{}.txt".format(os.path.basename(srcFile).split('.')[0], funName), 'w+')
    #         sys.stdout = outFile
    #
    #         fun = eval(funName) # 通过字符串调用函数
    #         selectProperRadius(0, 5, 0.05, fun)  # 步长设置为0.05
    #         stdOut.write("{}数据集, {}函数 测试完毕!!!\n".format(DATA_PATH, funName)) # 通过标准输出表示测试到那里了


    '''
        举例构造
    '''
    # print("######################################################")
    # disMatrix = generateDisMatrix(conditionData, [0,1,2,3])
    # neighbors = generateNeighbor(disMatrix, radius)
    # print("生成的邻域为:\n",neighbors)
    # getPos(decisionClasses, neighbors)





