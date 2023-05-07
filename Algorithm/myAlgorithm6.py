'''
2023-04-05 10:33:17
'''
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def generateDecisionClasses(Y: np.ndarray) -> dict:
    '''
    :param Y: 数据集的决策属性部分 值得注意的是将数据集的格式进行了统一 决策属性是从1开始的
    :return: 各个决策类的集合
    '''
    decClasses = dict()
    decValues = np.unique(Y)
    for decValue in decValues:
        decClasses[decValue] = set(np.where(Y == decValue)[0])
    return decClasses


# region 根据OD值对条件属性打分排序的逻辑
def getTwoSectionIntersectAndUnion(section1: list[int], section2: list[int]) -> tuple:
    '''
    :param section1: 列表形式的区间1 是一个二维int数组
    :param section2: 列表形式的区间2 是一个二维int数组
    :return: 两个区间交集的长度 两个区间并集的长度
    '''

    # print("区间1为{}".format(section1))
    # print("区间2为{}".format(section2))
    left1 = section1[0]  # 区间1的左端点
    right1 = section1[1]  # 区间1的右端点
    left2 = section2[0]  # 区间2的左端点
    right2 = section2[1]  # 区间2的右端点

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
    '''
    :param X: 数据集的条件属性部分
    :param attr: 将要进行打分的属性
    :param decClasses: 决策类
    :return: 对该属性的CD值 DIS值
    '''
    n = len(decClasses)
    DiMinArr = [0] * n
    CaDiArr = [0] * n
    DiMaxArr = [0] * n

    for i in range(n):  # 获取某一个决策类在该属性下的取值情况
        if (i+1) in decClasses:
            Di = decClasses[i + 1] # 获取该决策类下的所有索引情况
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
    return CDValue, DISValue


def sortAttrByOd(X: np.ndarray, decClasses: dict)->list[float]:
    '''
    :param X: 数据集的条件属性部分
    :param decClasses: 根据决策属性划分的决策类
    :return: 对于条件属性来说的 各个属性的得分情况
    '''
    attrNum = X.shape[1]
    CDScores = [0] * attrNum
    DISScores = [0] * attrNum
    odScores = [0] * attrNum

    for i in range(attrNum):  # 遍历每一个属性计算其od得分
        CDValue, DISValue = getOneAttrOdAndDisValue(X, [i], decClasses)
        CDScores[i] = CDValue
        DISScores[i] = DISValue
        odScores[i] = (CDValue+0.01) / (DISValue+0.01)
    return odScores  # 返回的是各个属性所对应的od的取值


def getRankedAttrsByOd(X:np.ndarray, Y:np.ndarray)->list[int]:
    '''
    :param X: 数据集的条件属性部分
    :param Y: 数据集的决策属性部分
    :return: 将所有的条件属性按照OD的取值进行排序
    '''
    decClasses = generateDecisionClasses(Y)
    odScores = sortAttrByOd(X, decClasses)  # 得到每一个属性的od得分
    sortedAttrs = np.argsort(odScores)
    return list(sortedAttrs)

def sortAttrByMulti(X:np.ndarray, Y:np.ndarray, proportions):
    sampleNum, attrNum = X.shape
    multiScores = [0] * attrNum
    clusterNums, granuleSampleNumThresholds = getGranules(sampleNum, proportions)

    # _, _, testScore = getMultiGranleClusterByKeamns(X, Y, list([3, 5, 9, 10, 13]), clusterNums,granuleSampleNumThresholds)
    # print(testScore)
    # exit()
    for i in range(attrNum):
        _, _, multiScores[i] = getMultiGranleClusterByKeamns(X, Y, list([i]), clusterNums,
                                                                            granuleSampleNumThresholds)
    # print(multiScores)
    # exit()
    return multiScores


def getRankedAttrsByMulti(X:np.ndarray, Y:np.ndarray, proprotions)->list[int]:
    multiScores = sortAttrByMulti(X, Y, proprotions)
    sortedAttrs = np.argsort(multiScores)
    return list(sortedAttrs)

# endregion

# region 针对某个条件属性集 通过初次聚类 多次合并的方式生成多个粒度下的聚类结果
def getGranules(sampleNum: int, proportions: list[int] = [0.025, 0.05, 0.075, 0.1, 0.125]):
    '''
    :param sampleNum:
    :param proportions:
    :return: 根据各个比例 返回各个簇的初始簇数 和 最终各个簇应该满足的样本个数
    '''
    n = len(proportions)
    granuleSampleNumThresholds = [0] * n
    clusterNums = [0] * n
    for i, proportion in enumerate(proportions):
        granuleSampleNumThresholds[i] = int(sampleNum * proportion)
        clusterNums[i] = sampleNum // granuleSampleNumThresholds[i] + 1
    return clusterNums, granuleSampleNumThresholds

def getClustersAndSampleNums(clusterLabels, clusterNum):
    '''
    :param clusterLabels: 经过Kmenas之后 每个样本所属于的簇的索引
    :param clusterNum: 簇的数量
    :return: 各个簇中样本是什么情况 有哪些样本吧 以及 簇中的样本数量各是多少
    '''
    clusters = []
    clusterSampleNums = []
    for i in range(clusterNum):
        cluster = np.where(clusterLabels == i)[0]
        clusters.append(cluster)
        clusterSampleNums.append(len(cluster))
    return clusters, clusterSampleNums

def mergeTwoCluster(clusters, clusterSampleNums, clusterIdx1, clusterIdx2):
    '''
    :param clusters: 合并前的聚类情况
    :param clusterSampleNums: 合并前聚类中各个簇中样本数量
    :param clusterIdx1: 将要合并的第一个簇的索引值
    :param clusterIdx2: 将要合并的第二个簇的索引值
    :return:
    '''
    clusters[clusterIdx2] = np.append(clusters[clusterIdx1], clusters[clusterIdx2])
    clusterSampleNums[clusterIdx2] = len(clusters[clusterIdx2])
    clusters = np.delete(clusters, clusterIdx1)
    clusterSampleNums = np.delete(clusterSampleNums, clusterIdx1)
    return clusters, clusterSampleNums

def twoClusterDistance(cluster1: np.ndarray, cluster2: np.ndarray, x: np.ndarray):
    '''
    :param cluster1: 要进行计算距离的第一个簇
    :param cluster2: 要进行计算距离的第二个簇
    :param X: 用于计算簇距离的样本数据
    :return: 两个簇之间距离 以及决定簇距离的两个样本
    '''
    minDis = 100
    idx1 = 0
    idx2 = 0
    for sample1 in cluster1:
        for sample2 in cluster2:
            curDis = np.linalg.norm(x[sample1] - x[sample2])
            if curDis < minDis:
                minDis = curDis
                idx1 = sample1
                idx2 = sample2
    return minDis, idx1, idx2

def mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold, X, Y):
    '''
    :param clusters: 目前的簇情况
    :param clusterSampleNums:
    :param granuleSampleNumThreshold:
    :param y:
    :return:
    '''
    count = 0
    preClusterSampleNums = []

    while np.any(clusterSampleNums < granuleSampleNumThreshold):  # 每次合并一对簇 只要聚类结果中存在一个簇中的样本少于阈值就继续进行合并
        if count != 0 and len(preClusterSampleNums) == len(clusterSampleNums):  # 无法再对簇进行融合了 无法找到两个簇 决定两个簇之间距离的两个样本的标签可能不一致 一个尽量的原则 在尽可能对原有粒度进行改变的情况下 尊重数据本身的分布情况
            # print("无法删减跳出")
            break
        preClusterSampleNums = clusterSampleNums

        for i in range(len(clusters)):
            if clusterSampleNums[i] >= granuleSampleNumThreshold:  # 该簇中样本数量已经满足要求
                continue

            mergeDepInfo = []  # 做出簇合并决策的依据
            # 对该簇与其他各个簇之间的距离信息进行记录
            for j in range(len(clusters)):
                if j == i:
                    continue

                info = [0] * 5  # 0,1 指明是哪两个簇之间的信息 2,3 两个簇之间距离最近的样本索引 4 距离
                info[0], info[1] = i, j
                info[4], info[2], info[3] = twoClusterDistance(clusters[i], clusters[j], X)
                # 这里可以根据已有的信息算一个得分 可以设置一个特殊条件如果满足特殊条件则直接进行合并
                mergeDepInfo.append(info)

            # 根据mergeDepInfo中的信息进行簇合并
            mergeDepInfo.sort(key=lambda x: x[4])  # 按照距离进行排序
            # for k in range(len(mergeDepInfo)-1,0,-1):
            for k in range(len(mergeDepInfo)):
                if Y[mergeDepInfo[k][2]] == Y[mergeDepInfo[k][3]]:  # 只有产生两个簇之间距离的两个样本标记相同才进行合并
                    # print(mergeDepInfo[k])
                    clusters, clusterSampleNums = mergeTwoCluster(clusters, clusterSampleNums,
                                                                mergeDepInfo[k][0],
                                                                mergeDepInfo[k][1])
                    break
            break
        # if count == 5: break # 对合并的轮数进行限制防止进入死循环
        count += 1
    return clusters, clusterSampleNums

# region 评估属性集得分相关的函数
def getYPred(y, clusters):
    '''
    :param y: 原始数据集的标签
    :param clusters: 各个簇中的样本的情况
    :return: 通过聚类给样本打的标记
    '''
    y_pred = np.array([0] * len(y))
    for cluster in clusters:
        y_pred[cluster] = [np.argmax(np.bincount(y[cluster])) for _ in range(len(cluster))]  # 对于一个簇来说 其中的一个样本都有一个标记 最多的那个标记 作为这个簇中所有样本的标记
    return y_pred


def getClusterPurity(labels_true, labels_pred):  # 聚类各个簇的纯度
    predExact = np.bincount(labels_true == labels_pred).max()
    return predExact / len(labels_true)

def getmultiClusterStrucScore(multiClusterStrucArr: list[np.ndarray], Y: np.ndarray):
    '''
    :param multiClusterStrucArr:
    :param Y:
    :return: 传入的是该条件属性集下的多粒度聚类结果 得到的是各个粒度下聚类结果的纯度的平均值
    '''
    n = len(multiClusterStrucArr)
    scores = [0] * n
    for i in range(n):
        yPred = getYPred(Y, multiClusterStrucArr[i])
        scores[i] = getClusterPurity(Y, yPred)
    finScore = sum(scores) / n
    # TODO: 这里可以选择直接将scores数组进行返回然后指定某一个属性集得分优于另一个属性集得分的策略
    return finScore
# endregion

def getMultiGranleClusterByKeamns(X: np.ndarray, Y: np.ndarray, attrSet: list[int], clusterNums: list[int],
                                granuleSampleNumThresholds: list[int]):
    '''
    :param X: 数据集条件属性部分
    :param Y: 数据集决策属性部分
    :param attrSet: 将要生成多粒度聚类结构的属性集
    :param clusterNums: 各个粒度下要求的聚类簇数
    :param granuleSampleNumThresholds: 各个粒度下簇中样本至少要大于这个数
    :return: 针对一个数据集 一部分(指定条件属性集合) 生成多个粒度下的聚类结果
    '''
    multiClusterStrucArr = []
    multiClusterSampleNumArr = []

    for i in range(len(clusterNums)):  # 针对一个属性集合在一个粒度下的得分 i代表第i+1个粒度下 各个属性集合的得分情况
        clusterNum = clusterNums[i]
        granuleSampleNumThreshold = granuleSampleNumThresholds[i]

        # 只进行一次聚类之后都是针对上一次的聚类结果进行簇合并
        if i == 0:
            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(X[:, attrSet])
            clusterLabels = kmeans.labels_  # 记录的是每一个样本被划分到每一个簇的情况
            clusters, clusterSampleNums = getClustersAndSampleNums(clusterLabels, clusterNum)

        # 对部分簇进行合并使得每个簇里面的样本个数符合要求
        clusters = np.array(clusters)
        clusterSampleNums = np.array(clusterSampleNums)
        clusters, clusterSampleNums = mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold,
                                                    X[:, attrSet], Y)

        multiClusterStrucArr.append(clusters.copy())
        multiClusterSampleNumArr.append(clusterSampleNums.copy())

    multiClusterStrucScore = getmultiClusterStrucScore(multiClusterStrucArr, Y)
    return multiClusterStrucArr, multiClusterSampleNumArr, multiClusterStrucScore


def acceAttrRedByMultiGranleKMeans(X:np.ndarray, Y:np.ndarray, proportions: list[int] = [0.025, 0.05, 0.075, 0.1, 0.125])->set[int]:
    # 这里需要获取以下属性的排名 以便逐步加入约简集合中
    sampleNum, attrNum = X.shape # 这里可以直接初次kmeans直接从数据集中样本数量的1/3开始进行
    clusterNums, granuleSampleNumThresholds = getGranules(sampleNum, proportions)
    # print("对于各个聚类结果中的簇的数量为:{}, 要求各个簇中样本数量尽量不少于{}".format(clusterNums, granuleSampleNumThresholds))

    all = set(range(attrNum))
    _,_, allAttrMultiClusterStrucScore = getMultiGranleClusterByKeamns(X, Y, list(all), clusterNums, granuleSampleNumThresholds)
    print("全属性集下多粒度聚类结构的得分为{}".format(allAttrMultiClusterStrucScore))
    # rankedAttrs = getRankedAttrsByOd(X, Y) # 排序之后的属性的list
    rankedAttrs = getRankedAttrsByMulti(X, Y, proportions) # 排序之后的属性的list
    start_time = time.time()  # 程序开始时间

    # 获得了排序了的属性之后 现进行一些属性的预选择
    red = set()
    red.add(rankedAttrs[0]) # 默认会添加一个属性
    curScore = 0
    for attr in rankedAttrs[1: ]: # 这里相当于从第二个属性开始往约简集里加
        multiClusterStrucArr, multiClusterSampleNumArr, multiClusterStrucScore = getMultiGranleClusterByKeamns(X, Y, list(red|set([attr])), clusterNums, granuleSampleNumThresholds)
        if multiClusterStrucScore>curScore:
            red.add(attr)
            curScore = multiClusterStrucScore
        print("当前属性集合{}下的多粒度聚类结构的得分为{}".format(red, curScore))

    # 删除多余属性的过程
    prepareDelAttr = list(red)
    finScore = curScore
    for attr in prepareDelAttr:
        multiClusterStrucArr, multiClusterSampleNumArr, multiClusterStrucScore = getMultiGranleClusterByKeamns(X, Y, list(red-set([attr])), clusterNums, granuleSampleNumThresholds)
        if multiClusterStrucScore>finScore:
            finScore = multiClusterStrucScore
            red = red-set([attr])
            if len(red)==1: break

    end_time = time.time()  # 程序结束时间
    run_time_sec = end_time - start_time  # 程序的运行时间，单位为秒
    return red, finScore, run_time_sec

# region 用于查看某一个属性集下的多粒度聚类结构
def showResultUnderAttrSet(multiClusterStrucArr, multiClusterSampleNumArr, multiClusterStrucScore):
    n = len(multiClusterStrucArr)
    for i in range(n):
        print("第{}个粒度下的聚类结果为".format(i+1))
        for cu in multiClusterStrucArr[i]: # 该结果下各个簇的情况如下
            print(cu)
    print("===================================================分割线===================================================")
    for i in range(n):
        print("第{}个粒度下的聚类结果中各个簇中的样本数量为".format(i + 1))
        print(multiClusterSampleNumArr[i])
    print("===================================================分割线===================================================")
    print("该多粒度聚类结构的得分为{}".format(multiClusterStrucScore))
# endregion


if __name__ == "__main__":
    dataSet = ['fertility_Diagnosis', 'BreastTissue', 'Iris', 'wine', 'plrx',
                'GlaucomaM', 'Sonar', 'seeds', 'Glass', 'accent',
                'PimaIndiansDiabetes', 'Ionosphere', 'movement', 'vote', 'musk',
                'wdbc', 'diamonds_filter', 'australian', 'BreastCancer', 'diabetes',
                'pima', 'College', 'Vehicle', 'german', 'data_banknote', 'waveform']

    path = '../DataSet_TEST/ori/{}.csv'.format("iris")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    Y = Y.astype(int)

    red, finScore, runTime = acceAttrRedByMultiGranleKMeans(X, Y)
    print("最终的属性约简为:{} 得分为{} 运行时间为{}".format(red, finScore, runTime))
