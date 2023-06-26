import heapq

import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def getTwoSectionIntersectAndUnion(section1, section2):
    '''
    :param section1: 列表形式的区间1
    :param section2: 列表形式的区间2
    :return: 两个区间交集的长度 两个区间并集的长度
    '''
    # print("区间1为{}".format(section1))
    # print("区间2为{}".format(section2))
    left1 = section1[0]
    right1 = section1[1]
    left2 = section2[0]
    right2 = section2[1]

    endPoints = sorted([left1, right1, left2, right2])

    interSect = []
    up = 0  # 两个区间的交集的长度
    down = 0  # 两个区间的并集的长度
    flag = False
    for i in range(4):
        if left1 <= endPoints[i] <= right1 and left2 <= endPoints[i] <= right2:
            interSect.append(endPoints[i])
            flag = True

    if flag:  # 如果两者有交集的话
        up = interSect[1] - interSect[0]
        down = endPoints[-1] - endPoints[0]
    else:  # 如果两个区间没有交集
        down = (section1[1] - section1[0]) + (section2[1] - section2[0])
    # print("交集的长度为{}, 并集的长度为{}".format(up, down))
    return up, down


def getOneAttrOdAndDisValue(X: np.ndarray, attr: int, decClasses: dict):
    n = len(decClasses)
    DiMinArr = [0] * n
    CaDiArr = [0] * n
    DiMaxArr = [0] * n

    for i in range(n):  # 获取某一个决策类在该属性下的取值情况
        Di = decClasses[i + 1]
        DiMinArr[i] = np.min(X[list(Di), [attr]])
        DiMaxArr[i] = np.max(X[list(Di), [attr]])
        CaDiArr[i] = np.mean(X[list(Di), [attr]])

    CDValue = 0
    DISValue = 0
    # 遍历任意两个决策类计算CD值和DIS值
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            Dim = DiMinArr[i]
            DiM = DiMaxArr[i]
            Djm = DiMinArr[j]
            DjM = DiMaxArr[j]

            up, down = getTwoSectionIntersectAndUnion([Dim, DiM], [Djm, DjM])

            CaDi = CaDiArr[i]
            CaDj = CaDiArr[j]
            fmDiDj = min(Dim, Djm)
            fMDiDj = max(DiM, DjM)
            DISValue += np.abs(CaDj - CaDi) / (fMDiDj - fmDiDj)
            CDValue += up / down if down != 0 else (up + 0.01) / (down + 0.01)
    # print(CDValue, DISValue)
    return CDValue, DISValue


def sortAttrByOd(X: np.ndarray, decClasses: dict):
    attrNum = X.shape[1]
    CDScores = [0] * attrNum
    DISScores = [0] * attrNum
    odScores = [0] * attrNum

    for i in range(attrNum):  # 遍历每一个属性计算其od得分
        CDValue, DISValue = getOneAttrOdAndDisValue(X, [i], decClasses)
        CDScores[i] = CDValue
        DISScores[i] = DISValue
        odScores[i] = CDValue / DISValue
    return odScores


def getKnearestSituationUnderOneAttr(X: np.ndarray, attr: int, K: int):
    sampleNum, attrNum = X.shape
    ori = X[:, attr].reshape(1, -1)[0]
    print(ori)
    knearestRelation = dict()
    # 找离每个样本最近的K个邻居
    for i in range(sampleNum):
        arr = np.abs(ori - ori[i])
        knearestRelation[i] = np.argsort(arr)[:K]
    print(knearestRelation)


def reductionByOdAndKnearest(X: np.ndarray, Y: np.ndarray, K: int):
    red = None
    decClasses = generateDecisionClasses(Y)
    # getOneAttrOdAndDisValue(X, [0], decClasses)
    odScores = sortAttrByOd(X, decClasses)  # 得到每一个属性的od得分
    sortedAttrs = np.argsort(odScores)
    print(sortedAttrs)

    return red


if __name__ == "__main__":
    # print("你好世界")
    path = '../DataSet_TEST/{}.csv'.format("wine")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]

    # 对Y进行处理 将决策属性的每一个值都转化为 1, 2, 3, 4...
    uniqueVal = np.unique(Y)
    for i in range(len(uniqueVal) - 1, -1, -1):
        Y[Y == uniqueVal[i]] = i + 1
    Y = Y.astype(int)

    # red = reductionByOdAndKnearest(X, Y, 5)
    # print(red)
    # print(Y)

    getKnearestSituationUnderOneAttr(X, [0], 3)
