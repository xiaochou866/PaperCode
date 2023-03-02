import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


# region 评估属性相关函数
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
    # print(scores)
    finScore = sum(scores) / n
    # TODO: 这里可以选择直接将scores数组进行返回然后指定某一个属性集得分优于另一个属性集得分的策略
    return finScore
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


def getMultiGranleClusterByKeamns(X: np.ndarray, Y: np.ndarray, attrSet: list[int], clusterNums: list[int],
                                granuleSampleNumThresholds: list[int]):
    '''
    :param X:
    :param Y:
    :param attrSet:
    :param clusterNums:
    :param granuleSampleNumThresholds:
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
    return multiClusterStrucArr, multiClusterSampleNumArr
# endregion

# region 属性集变化调整样本的分配从而调整聚类结果中的各个簇
def checkTwoClusterEquals(cluster1: np.ndarray, cluster2: np.ndarray) -> bool:
    '''
    :param cluster1: 前一种聚类中的一个簇
    :param cluster2: 后一种聚类结果中的一个簇
    :return: 只有当这两个簇中的样本数量一致并且相同的时候返回true
    '''
    if len(cluster1) != len(cluster2):
        return False
    cluster1.sort()
    cluster2.sort()
    return (cluster1 == cluster2).all()


def checkTwoClustersEquals(clusters1: list[np.ndarray], clusters2: list[np.ndarray]) -> bool:
    if len(clusters1) != len(clusters2):
        return False
    n = len(clusters1)
    for i in range(n):
        if not checkTwoClusterEquals(clusters1[i], clusters2[i]):
            return False
    return True

def generateCentroids(X: np.ndarray, attrSet: list[int], clusters: np.ndarray):
    '''
    :param X: 原始数据集在条件属性上的取值
    :param clusters: 各个簇中的样本情况
    :param attrSet: 考虑生成各个样本质心的属性集合
    :return:
    '''
    tempX = X[:, attrSet]
    centroids = []
    for cluster in clusters:
        clusterVector = tempX[cluster, :]  # 取出各个簇中样本在对应属性集上的向量
        centroids.append(np.mean(clusterVector, 0))
    return centroids


def calDisSampleToCentroid(X: np.ndarray, attrSet: list[int], centroids: np.ndarray):
    '''
    :param X: 原始数据集的条件属性部分
    :param attrSet: 参与计算的属性集合
    :param centroids: 上一次的各个簇的质心的情况
    :return: 数据集在属性集上各个样本到质心的距离 之后用于选择一个最小的质心重新及逆行分配
    '''
    m = len(X)
    n = len(centroids)
    disSampleToCentroid = np.zeros((m, n))

    partX = X[:, attrSet]
    for i in range(m):
        for j in range(n):
            disSampleToCentroid[i][j] = np.sum(np.square(partX[i, :] - centroids[j]))
    return disSampleToCentroid

def disBetweenTwoCentroids(centroids1, centroids2) -> float:
    '''
    :param centroids1: 前一种聚类结果各个簇的簇心
    :param centroids2: 后一种聚类结果各个簇的簇心
    :return:
    '''
    # print(centroids1)
    # print(centroids2)
    diff = np.array(centroids1) - np.array(centroids2)
    # print(diff)
    diffVal = sum([sum(np.square(e)) for e in diff])
    return diffVal

def KmeansChangeByAttrSet(X: np.ndarray, Y: np.ndarray, oldclusters: list[np.ndarray], attrSet: list[int],
                        maxIterator: int = 20, diffValue=0.2):
    centroids = generateCentroids(X, attrSet, oldclusters)
    specialModel = KMeans(n_clusters=len(centroids), init=centroids)  # 'init': 'k-means++',默认为k-means++
    clusRes = specialModel.fit(X[:,attrSet])
    newClusters, newClusterSampleNums = getClustersAndSampleNums(clusRes.labels_, len(centroids))
    return newClusters, newClusterSampleNums

def getCurClusterResult(X: np.ndarray, y: np.ndarray, attrSet: list[int], preMultiClusterStrucArr: list[np.ndarray]):
    curMultiClusterStrucArr = []
    curMultiClusterSampleNumArr = []

    for clusterStru in preMultiClusterStrucArr:
        newClusters, newClusterSampleNums = KmeansChangeByAttrSet(X, y, clusterStru, attrSet)
        curMultiClusterStrucArr.append(newClusters.copy())
        curMultiClusterSampleNumArr.append(newClusterSampleNums.copy())

    return curMultiClusterStrucArr, curMultiClusterSampleNumArr
# endregion

def generateInitMultiCluster(X:np.ndarray, Y:np.ndarray, attrNum:int, clusterNums:list[int], granuleSampleNumThresholds:list[int]):
    maxMultiClusterStrucScore = 0
    selectedAttr = 0
    initMultiClusterStrucArr = None
    initMultiClusterSampleNumArr = None

    for i in range(attrNum):
        tmpMultiClusterStrucArr, tmpMultiClusterSampleNumArr = getMultiGranleClusterByKeamns(X, Y, [i], clusterNums,
                                                                                            granuleSampleNumThresholds)
        tmpMultiClusterStrucScore = getmultiClusterStrucScore(tmpMultiClusterStrucArr, Y)
        if tmpMultiClusterStrucScore > maxMultiClusterStrucScore:
            maxMultiClusterStrucScore = tmpMultiClusterStrucScore
            selectedAttr = i
            initMultiClusterStrucArr = tmpMultiClusterStrucArr.copy()
            initMultiClusterSampleNumArr = tmpMultiClusterSampleNumArr.copy()

    return selectedAttr, maxMultiClusterStrucScore, initMultiClusterStrucArr, initMultiClusterSampleNumArr

def forwardAttrRedByMultiGranleKMeans(X: np.ndarray, Y: np.ndarray):
    red = None
    sampleNum, attrNum = X.shape
    clusterNums, granuleSampleNumThresholds = getGranules(sampleNum)  # 生成多个粒度用于后序生成多个粒度下的聚类结果 [45, 23, 14, 11, 9] [4, 8, 13, 17, 22]

    selectedAttr, maxMultiClusterStrucScore, initMultiClusterStrucArr, initMultiClusterSampleNumArr = generateInitMultiCluster(X, Y, attrNum, clusterNums, granuleSampleNumThresholds)

    # centroids = generateCentroids(X, [selectedAttr, 1], initMultiClusterStrucArr[0])
    # specialModel = KMeans(n_clusters=len(centroids), init=centroids)  # 'init': 'k-means++',默认为k-means++
    # clusRes = specialModel.fit(X[:,[selectedAttr, 1]])
    # print(clusRes.labels_)
    # exit()

    A = set([selectedAttr])
    AT = set(range(attrNum))
    preMultiClusterStrucArr = initMultiClusterStrucArr
    preScore = maxMultiClusterStrucScore

    while True:
        curScore = -100
        selectedAttr = 0
        curMultiClusterStrucArr = None
        curMultiClusterSampleNumArr = None
        for a in AT - A:  #
            # print(A | set([a]))
            tmpMultiClusterStrucArr, tmpMultiClusterSampleNumArr = getCurClusterResult(X, Y, list(A | set([a])),
                                                                                    preMultiClusterStrucArr)
            tmpScore = getmultiClusterStrucScore(tmpMultiClusterStrucArr, Y)

            if tmpScore > curScore:
                curScore = tmpScore
                selectedAttr = a
                curMultiClusterStrucArr = tmpMultiClusterStrucArr
                curMultiClusterSampleNumArr = tmpMultiClusterSampleNumArr
        if curScore < preScore:
            break
        preScore = curScore
        preMultiClusterStrucArr = curMultiClusterStrucArr
        preMultiClusterSampleNumArr = curMultiClusterSampleNumArr
        A = A | set([selectedAttr])

    return A


if __name__ == "__main__":
    dataSet = ["iris", "wine", "Glass", "plrx", "wdbc"]
    path = '../DataSet_TEST/{}.csv'.format("iris")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-2]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    Y = Y.astype(int)

    red = forwardAttrRedByMultiGranleKMeans(X, Y)
    print(red)
