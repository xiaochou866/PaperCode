import warnings
import numpy as np
from sklearn.metrics import pair_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import datetime
warnings.filterwarnings('ignore')


def twoClusterDistance(cluster1: np.ndarray, cluster2: np.ndarray):
    '''
    :param cluster1:
    :param cluster2:
    :return: 两个簇之间的距离 取任两个样本之间的最小值
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


def mergeTwoCluster(clusters, clusterSampleNums, clusterIdx1, clusterIdx2):
    # 将前一个簇里的样本合并到后一个簇 并对簇中的样本个数进行更新
    clusters[clusterIdx2] = np.append(clusters[clusterIdx1], clusters[clusterIdx2])
    clusterSampleNums[clusterIdx2] = len(clusters[clusterIdx2])
    clusters = np.delete(clusters, clusterIdx1)
    clusterSampleNums = np.delete(clusterSampleNums, clusterIdx1)
    return clusters, clusterSampleNums


def getClusterPurity(labels_true, labels_pred):  # 聚类各个簇的纯度
    predExact = np.bincount(labels_true == labels_pred).max()
    return predExact / len(labels_true)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):  # RI ARI F 兰德系数 调整兰德系数 F值
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    # ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    # p, r = tp / (tp + fp), tp / (tp + fn)
    # f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    # return ri, ari, f_beta
    return ri


def getYPred(y, clusters):
    '''
    :param y: 原始数据集的标签
    :param clusters: 各个簇中的样本的情况
    :return: 通过聚类给样本打的标记
    '''
    y_pred = np.array([0] * len(y))
    for cluster in clusters:
        y_pred[cluster] = [np.argmax(np.bincount(y[cluster])) for _ in range(len(cluster))] # 对于一个簇来说 其中的一个样本都有一个标记 最多的那个标记 作为这个簇中所有样本的标记
    return y_pred


def getGranules(sampleNum: int, proportions: list[int]):
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


def generateDecisionClasses(Y: np.ndarray) -> list[np.ndarray]:
    '''
    :param Y: 数据集决策属性那一列
    :return:  U/D
    '''
    U_D = []
    lables = np.unique(Y)
    for lable in lables:
        U_D.append(np.where(Y == lable)[0])
    return U_D


def mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold):
    count = 0
    preclusterSampleNums = []

    while np.any(clusterSampleNums < granuleSampleNumThreshold):  # 每次合并一对簇

        # print("使得每一个簇中样本数量不小于{}".format(granuleSampleNumThreshold))
        # print(clusterSampleNums)
        # print(np.any(clusterSampleNums < granuleSampleNumThreshold))

        if count != 0 and len(preclusterSampleNums) == len(clusterSampleNums):  # 无法再对簇进行融合了
            # print("无法删减跳出")
            break
        preclusterSampleNums = clusterSampleNums

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
                info[4], info[2], info[3] = twoClusterDistance(clusters[i], clusters[j])
                # 这里可以根据已有的信息算一个得分 可以设置一个特殊条件如果满足特殊条件则直接进行合并
                mergeDepInfo.append(info)

            # 根据mergeDepInfo中的信息进行簇合并
            mergeDepInfo.sort(key=lambda x: x[4])  # 按照距离进行排序
            # for k in range(len(mergeDepInfo)-1,0,-1):
            for k in range(len(mergeDepInfo)):
                if y[mergeDepInfo[k][2]] == y[mergeDepInfo[k][3]]:  # 只有产生两个簇之间距离的两个样本标记相同才进行合并
                    # print(mergeDepInfo[k])
                    clusters, clusterSampleNums = mergeTwoCluster(clusters, clusterSampleNums,
                                                                    mergeDepInfo[k][0],
                                                                    mergeDepInfo[k][1])
                    break
            break
        # if count == 5: break # 对合并的轮数进行限制防止进入死循环
        count += 1
    return clusters, clusterSampleNums


def getPosValue(variablePrecision, clusters, y, labels):
    posSampleNum = 0
    for cluster in clusters:
        # cluster [  0   3   5   6   7   8   9  10  11  12...
        n = len(cluster)
        clusterLabel = y[cluster]
        for label in labels:
            # print("标签为{}".format(label))
            samples = np.where(clusterLabel == label)[0]
            if len(samples) / n >= variablePrecision:
                posSampleNum += len(samples)
    return posSampleNum / len(y)


def getFuseScore(scores, weights):
    '''
    :param scores: 各个粒度下的得分
    :param weights: 各个粒度得分所占的权重
    :return: fused measure
    '''
    fuseScore = 0
    sumWeight = sum(weights)
    weights = [e / sumWeight for e in weights]
    for i in range(len(scores)):
        fuseScore += scores[i] * weights[i]
    return fuseScore


def getAttrSetScore(x:np.ndarray, y:np.ndarray, attrSet:set[int]):
    '''
    :param x: 数据集的条件属性部分
    :param y: 数据集的决策属性部分
    :param attrSet: 代进行评估的属性集合
    :return: 该属性集合的得分
    '''

    x = x[:, attrSet]

    # 不同的粒度
    # proportions = [0.05, 0.075, 0.1, 0.125, 0.15]  # 按照数据集样本数的比例划分不同的粒度
    # proportions = [0.02, 0.04, 0.06, 0.08, 0.1]  # 按照数据集样本数的比例划分不同的粒度
    proportions = [0.025, 0.05, 0.075, 0.1, 0.125]  # 按照数据集样本数的比例划分不同的粒度

    # 从x 和 y中获取下面程序所需要的信息
    proportionNum = len(proportions)
    sampleNum = len(x)  # 数据集的样本总数
    labels = np.unique(y)  # 所有的不同的标签

    # 获取对于各个簇的要求
    clusterNums, granuleSampleNumThresholds = getGranules(sampleNum, proportions)

    # 现在针对全属性集计算出一个得分 现在想要做的事情
    scores = [0] * proportionNum
    weights = [0] * proportionNum

    for i in range(proportionNum):  # 针对一个属性集合在一个粒度下的得分 i代表第i+1个粒度下 各个属性集合的得分情况
        clusterNum = clusterNums[i]
        granuleSampleNumThreshold = granuleSampleNumThresholds[i]

        # 聚类 每次粒度都重新聚类
        # kmeans = KMeans(n_clusters=clusterNum, random_state=0,max_iter=10).fit(x)
        # clusterLabels = kmeans.labels_
        # clusters, clusterSampleNums = getClustersAndSampleNums(clusterLabels)

        # 只进行一次聚类之后都是针对上一次的聚类结果进行簇合并
        if i == 0:
            kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(x)
            clusterLabels = kmeans.labels_ # 记录的是每一个样本被划分到每一个簇的情况
            clusters, clusterSampleNums = getClustersAndSampleNums(clusterLabels, clusterNum)

        # 对部分簇进行合并使得每个簇里面的样本个数符合要求
        clusters = np.array(clusters)
        clusterSampleNums = np.array(clusterSampleNums)
        clusters, clusterSampleNums = mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold)


        # region 查看每一个合并后 查看最终该粒度下的簇的情况
        print("簇数为:{}, 最终的聚类结果各个簇中的样本数量尽量不少于{}".format(clusterNum, granuleSampleNumThreshold))
        print("各个簇中的样本情况:{}".format(clusters))  # 聚类并和并簇之后 各个簇里面的样本情况
        print("各个簇中的样本数量情况:{}".format(clusterSampleNums))  # 聚类并和并簇之后 各个簇中的样本个数
        # endregion


        # 根据聚类结果得到预测的标签
        yPred = getYPred(y, clusters)

        # scores[i] = getPosValue(0.7, clusters, y, labels) # 根据各个标记的样本占该簇的比例来将样本归为正域
        scores[i] = getClusterPurity(y, yPred)

        # weights[i] = getClusterPurity(y, yPred)
        weights[i] = get_rand_index_and_f_measure(y, yPred)

    # print("各个粒度下该属性集的得分为{}".format(scores))
    # print("各个粒度下该属性集的得分的权重为{}".format(weights))
    finScore = getFuseScore(scores, weights)
    return finScore

# region 之前的版本 通过指定数据集的路径运行函数
def multiReductByKMeansTest(path):
    global x
    global y

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, :-1]
    x = MinMaxScaler().fit_transform(x)  # 归一化取值均归为0-1之间
    conditionAttrNum = x.shape[1]

    y = data[:, -1]
    y = y.astype(np.int)

    # exit()
    AT = set(range(conditionAttrNum))  # 全体属性集合
    A = set()  # 用于记录最终结果

    starttime = datetime.datetime.now()
    preScore = -100
    while True:
        T = AT - A  # 除去已经选择的属性
        curScore = -100 # 记录从这次的备选属性集合中选出来属性加入集合后的得分
        selectedAttr = 0  # 记录本次选出来的属性
        for i, a in enumerate(T):
            print("现在将要进行评估的属性集合为:{}".format(A | set([a])))
            tmpScore = getAttrSetScore(x, y, list(A | set([a])))
            if tmpScore>curScore:
                curScore = tmpScore
                selectedAttr = a

        # print("现在属性集合得分为:", curScore)
        if curScore<=preScore:
            break

        A.add(selectedAttr)
        preScore = curScore
        print(curScore)

    # long running
    endtime = datetime.datetime.now()
    runTime = (endtime - starttime).seconds

    return A, preScore, runTime
# endregion


def multiReductByKMeans(foldX, foldy):
    global x
    global y

    # data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = foldX
    # x = MinMaxScaler().fit_transform(x)  # 归一化取值均归为0-1之间
    conditionAttrNum = x.shape[1]

    y = foldy
    y = y.astype(np.int)

    AT = set(range(conditionAttrNum))  # 全体属性集合
    A = set()  # 用于记录最终结果

    starttime = datetime.datetime.now()

    preScore = -100
    while True:
        T = AT - A  # 除去已经选择的属性
        curScore = -100 # 记录从这次的备选属性集合中选出来属性加入集合后的得分
        selectedAttr = 0  # 记录本次选出来的属性
        for i, a in enumerate(T):
            tmpScore = getAttrSetScore(x, y, list(A | set([a])))
            if tmpScore>curScore:
                curScore = tmpScore
                selectedAttr = a

        # print("现在属性集合得分为:", curScore)
        if curScore<=preScore:
            break

        A.add(selectedAttr)
        preScore = curScore

    # long running
    endtime = datetime.datetime.now()
    runTime = (endtime - starttime).seconds

    return A, preScore, runTime # 属性约简结果 最终的属性集合的得分 运行时间

if __name__ == "__main__":
    path = "../DataSet_TEST/BreastCancer.csv"
    res = multiReductByKMeansTest(path)
    print(res)

# waveform.csv ({5, 6, 10, 14, 16}, 0.42295726600666916) 5000 2-3h
# wine.csv ({0, 1, 4, 6, 10, 12}, 0.9887005649717515)
# iris.csv ({0, 4}, 1.0)