import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pair_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# from Algorithm.myAlgorithm import getGranules, getClustersAndSampleNums, mergeClusters

def getFuseScore(scores, weights):
    '''
    :param scores: 各个粒度下的得分
    :param weights: 各个粒度得分所占的权重
    :return: fused measure
    '''
    # 方案1: 使用兰德系数做加权得分
    # fuseScore = 0
    # sumWeight = sum(weights)
    # weights = [e / sumWeight for e in weights]
    # for i in range(len(scores)):
    #     fuseScore += scores[i] * weights[i]
    #
    # 方案2: 使用得分的平均值
    fuseScore = sum(scores)/len(scores)
    return fuseScore

def getClusterPurity(labels_true, labels_pred):  # 聚类各个簇的纯度
    predExact = np.bincount(labels_true == labels_pred).max()
    return predExact / len(labels_true)

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):  # RI ARI F 兰德系数 调整兰德系数 F值
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    # ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    # p, r = tp / (tp + fp), tp / (tp + fn)
    # f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
    # return ri, ari, f_beta
    return ari

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids

def twoClusterDistance(cluster1: np.ndarray, cluster2: np.ndarray, x:np.ndarray):
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
    # 将前一个簇里的样本合并到后一个簇 并对簇中的样本个数进行更新
    clusters[clusterIdx2] = np.append(clusters[clusterIdx1], clusters[clusterIdx2])
    clusterSampleNums[clusterIdx2] = len(clusters[clusterIdx2])
    clusters = np.delete(clusters, clusterIdx1)
    clusterSampleNums = np.delete(clusterSampleNums, clusterIdx1)
    return clusters, clusterSampleNums

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

def mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold, x, y):
    '''
    :param clusters: 目前的簇情况
    :param clusterSampleNums:
    :param granuleSampleNumThreshold:
    :param y:
    :return:
    '''
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
                info[4], info[2], info[3] = twoClusterDistance(clusters[i], clusters[j], x)
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



def getOriClusterResult(X:np.ndarray, Y: np.ndarray, clusterNums, granuleSampleNumThresholds):
    '''
    :param X: 全属性集下的样本条件属性部分
    :param clusterNums:
    :param granuleSampleNumThresholds: 各个粒度下的样本数量要求
    :return:
    '''
    multiClusterStrucArr = []
    multiClusterSampleNumArr = []

    for i in range(len(clusterNums)):  # 针对一个属性集合在一个粒度下的得分 i代表第i+1个粒度下 各个属性集合的得分情况
        clusterNum = clusterNums[i]
        granuleSampleNumThreshold = granuleSampleNumThresholds[i]

        # 只进行一次聚类之后都是针对上一次的聚类结果进行簇合并
        if i == 0:
            kmeans = KMeans(n_clusters=clusterNum, random_state=None).fit(X)
            clusterLabels = kmeans.labels_ # 记录的是每一个样本被划分到每一个簇的情况
            clusters, clusterSampleNums = getClustersAndSampleNums(clusterLabels, clusterNum)

        # 对部分簇进行合并使得每个簇里面的样本个数符合要求
        clusters = np.array(clusters)
        clusterSampleNums = np.array(clusterSampleNums)
        clusters, clusterSampleNums = mergeClusters(clusters, clusterSampleNums, granuleSampleNumThreshold, X, Y)

        # print("第{}次聚类加簇合并之后的结果为:".format(i+1))
        # print(clusters)
        # print(clusterSampleNums)
        multiClusterStrucArr.append(clusters.copy())
        multiClusterSampleNumArr.append(clusterSampleNums.copy())
    return multiClusterStrucArr, multiClusterSampleNumArr


def generateCentroids(X: np.ndarray, clusters: np.ndarray, attrSet:list[int]):
    '''
    :param X: 原始数据集在条件属性上的取值
    :param clusters: 各个簇中的样本情况
    :param attrSet: 考虑生成各个样本质心的属性集合
    :return:
    '''
    tempX = X[:, attrSet]
    centroids = []
    for cluster in clusters:
        clusterVector = tempX[cluster, :] # 取出各个簇中样本在对应属性集上的向量
        centroids.append(np.mean(clusterVector, 0))
    return centroids

def calDisSampleToCentroid(X:np.ndarray, attrSet:list[int], centroids: np.ndarray):
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
            disSampleToCentroid[i][j] = np.sum(np.square(partX[i,:]-centroids[j]))
    return disSampleToCentroid

def checkTwoClusterEquals(cluster1: np.ndarray, cluster2: np.ndarray)->bool:
    '''
    :param cluster1: 前一种聚类中的一个簇
    :param cluster2: 后一种聚类结果中的一个簇
    :return: 只有当这两个簇中的样本数量一致并且相同的时候返回true
    '''
    if len(cluster1)!=len(cluster2):
        return False
    cluster1.sort()
    cluster2.sort()
    return (cluster1 == cluster2).all()

def checkTwoClustersEquals(clusters1:list[np.ndarray], clusters2: list[np.ndarray])->bool:
    if len(clusters1)!=len(clusters2):
        return False
    n = len(clusters1)
    for i in range(n):
        if not checkTwoClusterEquals(clusters1[i], clusters2[i]):
            return False
    return True

def disBetweenTwoCentroids(centroids1, centroids2)->float:
    '''
    :param centroids1: 前一种聚类结果各个簇的簇心
    :param centroids2: 后一种聚类结果各个簇的簇心
    :return:
    '''
    # print(centroids1)
    # print(centroids2)
    diff = np.array(centroids1)- np.array(centroids2)
    # print(diff)
    diffVal = sum([sum(np.square(e)) for e in diff])
    return diffVal

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


def getFusedScoreByMultiClusterResUnderAttrSet(multiClusterStrucArr:list[np.ndarray], y:np.ndarray):
    granuleNum = len(multiClusterStrucArr) # 记录粒度数
    # 现在针对全属性集计算出一个得分 现在想要做的事情
    scores = [0] * granuleNum
    weights = [0] * granuleNum
    for i in range(granuleNum):
        clusterStruc = multiClusterStrucArr[i]
        # print(clusterStruc)
        yPred = getYPred(y, clusterStruc)

        # scores[i] = getPosValue(0.7, clusters, y, labels) # 根据各个标记的样本占该簇的比例来将样本归为正域
        scores[i] = getClusterPurity(y, yPred)

        # weights[i] = getClusterPurity(y, yPred)
        weights[i] = get_rand_index_and_f_measure(y, yPred)

    print(scores)
    print(weights)

    finScore = getFuseScore(scores, weights)
    return finScore

def KmeansChangeByAttrSet(X:np.ndarray, Y:np.ndarray, oldclusters:list[np.ndarray], attrSet: list[int], maxIterator:int=20, diffValue=0.2):
    # print(type(attrSet))
    iteratorTime = 0

    # 不断调整 跳出的条件有三个: 1. 各个簇不发生变化 2. 质心之间的差异小于某个阈值 3. 达到最大迭代次数
    while iteratorTime<maxIterator:
        # print("进行了{}轮簇调整".format(iteratorTime+1))
        # 不断的将簇内的样本进行重新分配 当迭代次数达到给定值 或者 两次质心之间的欧式距离小于diffValue
        clusterNum = len(oldclusters)
        oldCentroids = generateCentroids(X, oldclusters, attrSet) # 计算前一次聚类结果中各个质心的情况
        disSampleToCentroid = calDisSampleToCentroid(X, attrSet, oldCentroids)  # 计算数据集中的各个样本到质心之间的距离

        sampleCluster = np.argmin(disSampleToCentroid, axis=1)  # 将每个样本所在的簇重新进行分配 即将样本分配到最近的簇中
        newClusters, newClusterSampleNums = getClustersAndSampleNums(sampleCluster, clusterNum)
        newCentroids = generateCentroids(X, newClusters, attrSet)

        diffVal = disBetweenTwoCentroids(oldCentroids, newCentroids)

        # print(diffVal)
        # if diffVal < 0.01 or checkTwoClustersEquals(oldclusters, newClusters):  # 当所有质心之间的差异和<0.01 直接跳出循环 不再进行调整
        #     break

        if checkTwoClustersEquals(oldclusters, newClusters):  # 当所有质心之间的差异和<0.01 直接跳出循环 不再进行调整
            break

        oldclusters = newClusters
        iteratorTime += 1
    return newClusters, newClusterSampleNums

def getCurClusterResult(X:np.ndarray, y:np.ndarray, attrSet:list[int], preMultiClusterStrucArr: list[np.ndarray]):
    curMultiClusterStrucArr = []
    curMultiClusterSampleNumArr = []

    for clusterStru in preMultiClusterStrucArr:
        newClusters, newClusterSampleNums = KmeansChangeByAttrSet(X, y, clusterStru, attrSet)
        curMultiClusterStrucArr.append(newClusters)
        curMultiClusterSampleNumArr.append(newClusterSampleNums)

    return curMultiClusterStrucArr, curMultiClusterSampleNumArr


def reductionByKMeansFormTopToBottom(X: np.ndarray, y:np.ndarray, proportions:list[float]=[0.025, 0.05, 0.075, 0.1, 0.125]):
    # 从x 和 y中获取下面程序所需要的信息
    proportionNum = len(proportions)
    sampleNum, attrNum = X.shape

    # 获取对于各个簇的要求
    clusterNums, granuleSampleNumThresholds = getGranules(sampleNum, proportions)
    preMultiClusterStrucArr, preMultiClusterSampleNumArr = getOriClusterResult(X, y, clusterNums, granuleSampleNumThresholds)
    preScore = getFusedScoreByMultiClusterResUnderAttrSet(preMultiClusterStrucArr, y)
    print(preScore)
    # exit()

    A = set(range(attrNum)) # 从全属性集开始进行约简
    while True:
        curScore = -100
        selectedAttr = 0
        curMultiClusterStrucArr = None
        curMultiClusterSampleNumArr = None
        for a in A: #
            tmpMultiClusterStrucArr, tmpMultiClusterSampleNumArr = getCurClusterResult(X, y, list(A-set([a])), preMultiClusterStrucArr)
            tmpScore = getFusedScoreByMultiClusterResUnderAttrSet(tmpMultiClusterStrucArr, y)
            print(list(A-set([a])))
            print(tmpScore)
            if tmpScore>curScore:
                curScore = tmpScore
                selectedAttr =  a
                curMultiClusterStrucArr = tmpMultiClusterStrucArr
                curMultiClusterSampleNumArr = tmpMultiClusterSampleNumArr
        if curScore<preScore:
            break
        preScore = curScore
        preMultiClusterStrucArr = curMultiClusterStrucArr
        preMultiClusterSampleNumArr = curMultiClusterSampleNumArr
        A = A-set([selectedAttr])

    return A

if __name__ == '__main__':
    path = '../DataSet_TEST/{}.csv'.format("accent")
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    Y = Y.astype(int)

    red = reductionByKMeansFormTopToBottom(X, Y)
    print(red)




    # # 先对一个大K值进行一次Kmeans 之后不断进行簇合并产生多个粒度
    # clusterNum = clusterNums[0]
    # kmeans = KMeans(n_clusters=clusterNums[0], random_state=0).fit(X)
    # clusterLabels = kmeans.labels_  # 记录的是每一个样本被划分到每一个簇的情况
    # clusters, clusterSampleNums = getClustersAndSampleNums(clusterLabels, clusterNum)
    # # print(clusters, clusterSampleNums)
    #
    # for i in range(len(clusterArr)):
    #     print("针对第{}个粒度下的聚类情况".format(i+1))
    #     print("聚类情况为:")
    #     print(clusterArr[i])
    #
    #     # centroids = generateCentroids(X, clusterArr[0], [0,1])
    #     # print("各个簇的中心为:", centroids)
    #
    #     newClusters, newClusterSampleNums = KmeansChangeByAttrSet(X, Y, clusterArr[i], [0, 1])
    #     print(clusterArr[i])
    #     print("聚类中各个簇的样本数量为:")
    #     print(clusterSampleNumArr[i])
    #
    #     print(newClusters)
    #     print("调整之后聚类中各个簇的样本数量为:")
    #     print(newClusterSampleNums)


