import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import tree,svm
from sklearn.neighbors import KNeighborsClassifier


# 将要进行实验的所有对比算法
from Algorithm.compareAlgorithm0 import reductionUseNeighborhoodRoughSet
from Algorithm.compareAlgorithm1 import reductionUseAttributeGroup
from Algorithm.compareAlgorithm2_1 import reductionUseDisSimilarity
from Algorithm.compareAlgorithm2_2 import reductionUseSimilarity
from Algorithm.compareAlgorithm3 import reductionUseWeightedNeighborhood
from Algorithm.compareAlgorithm4 import reductionUseVariableRadiusNeighborhoodRoughSet
from Algorithm.compareAlgorithm5 import reductionUseRandomSampling

def getXY(path: str):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
    Y = data[:, -1]
    Y = Y.astype(int)
    return X, Y

def getClassificationAccuracy(oriX, oriY, attrSet) -> list[float]:
    knn = KNeighborsClassifier(n_neighbors=10)
    clf = svm.SVC(kernel='linear', C=1)
    cart = tree.DecisionTreeClassifier(max_depth=5)
    score1 = cross_val_score(knn, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    score2 = cross_val_score(clf, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    score3 = cross_val_score(cart, oriX[:, list(attrSet)], oriY, cv=5, scoring='accuracy')  # 5折：交叉验证
    return score1, score2, score3

def generateRadiusArr(startRadius:float, endRadius:float, interval:float):
    '''
    :param startRadius: 实验的起始半径
    :param endRadius: 实验的终止半径
    :param interval: 半径之间的间隔
    :return: 用于实验的所有半径的arr
    '''
    radiusArr = []
    curRadius = startRadius
    while curRadius < endRadius:
        radiusArr.append(curRadius)
        curRadius += interval
    return radiusArr


# # 根据数据详情获取将要进行实验的所有数据集的名字的列表
# df = pd.read_excel("../DataSet_TEST/{}".format("数据详情-基于样本数升序.xlsx"))
# print(df["数据集"].tolist())


if __name__ == "__main__":
    # 展示将要进行实验的所有数据集
    dir = "../DataSet_TEST"
    dataCategory = ["ori", "1fold", "2fold", "3fold", "4fold", "5fold", "10noise", "20noise", "30noise"]
    dataSet = ['fertility_Diagnosis', 'BreastTissue', 'Iris', 'wine', 'plrx',
                'GlaucomaM', 'Sonar', 'seeds', 'Glass', 'accent',
                'PimaIndiansDiabetes', 'Ionosphere', 'movement', 'vote', 'musk',
                'wdbc', 'diamonds_filter', 'australian', 'BreastCancer', 'diabetes',
                'pima', 'College', 'Vehicle', 'german', 'data_banknote', 'waveform']
    # 设置将要进行实验的半径
    expRadius = generateRadiusArr(0.02, 0.42, 0.02) # 设置实验半径为 0.02, 0.04, ... ,0.4 这20个半径

    # 设置将要进行实验的算法
    # algorithmName = "NeighborhoodRoughSet" # 对应算法0
    # algorithmName = "AttributeGroupAttributeReduction" # 对应算法1
    algorithmName = "DisSimilarityAttributeReduction" # 对应算法2-1
    # algorithmName = "SimilarityAttributeReduction" # 对应算法2-2
    # algorithmName = "WeightedAttributeReduction" # 对应算法3
    # algorithmName = "VariableRadiusNeighborhoodRoughSet" # 对应算法4
    # algorithmName = "RandomSamplingAttribureReduction" # 对应算法5

    # algorithmName = "algorithm0" # 对应算法0
    # algorithmName = "algorithm1" # 对应算法1
    algorithmName = "algorithm2_1" # 对应算法2-1
    # algorithmName = "algorithm2_2" # 对应算法2-2
    # algorithmName = "algorithm3" # 对应算法3
    # algorithmName = "algorithm4" # 对应算法4
    # algorithmName = "algorithm5" # 对应算法5

    # 设置将要进行实验的数据种类
    # category = dataCategory[0]  # 指定数据的类别 每次专注于一种类别的数据集
    # 五折的数据集
    # category = dataCategory[1]  # 指定数据的类别 每次专注于一种类别的数据集
    category = dataCategory[2]  # 指定数据的类别 每次专注于一种类别的数据集
    # category = dataCategory[3]  # 指定数据的类别 每次专注于一种类别的数据集
    # category = dataCategory[4]  # 指定数据的类别 每次专注于一种类别的数据集
    # category = dataCategory[5]  # 指定数据的类别 每次专注于一种类别的数据集
    # 加了10 20 30噪声的数据集
    # category = dataCategory[6]  # 指定数据的类别 每次专注于一种类别的数据集
    # category = dataCategory[7]  # 指定数据的类别 每次专注于一种类别的数据集
    # category = dataCategory[8]  # 指定数据的类别 每次专注于一种类别的数据集


    # # 优先级 算法>数据集>半径
    for dataName in dataSet:
        # 原始数据
        oriPath = os.path.join(dir, "ori", "{}.csv".format(dataName))
        oriX, oriY = getXY(oriPath)
        sampleNum, attrNum = oriX.shape # 获取数据库中的样本数量以及属性数量


        # 指定的某一类别的数据
        path = os.path.join(dir, category, "{}.csv".format(dataName))
        X, Y = getXY(path)
        data = np.loadtxt(path, delimiter=",", skiprows=1)

        for radius in expRadius:
            # # 指定属性约简的算法
            # if algorithmName == "NeighborhoodRoughSet":
            #     red, score, runTime = reductionUseNeighborhoodRoughSet(X, Y, radius, "PRE") # 对应算法0 基于邻域粗糙集的属性约简
            # elif algorithmName == "AttributeGroupAttributeReduction":
            #     red, score, runTime = reductionUseAttributeGroup(data, radius, "POS", "PRE", X, Y) # 对应算法1 基于属性分组的属性约简
            # elif algorithmName == "DisSimilarityAttributeReduction":
            #     red, score, runTime = reductionUseDisSimilarity(data, radius, "POS", "PRE", X, Y) # 对应算法2-1 基于差异度的属性约简
            # elif algorithmName == "SimilarityAttributeReduction":
            #     red, score, runTime = reductionUseSimilarity(data, radius, "POS", "PRE", X, Y) # 对应算法2-2 基于相似度的属性约简
            # elif algorithmName == "WeightedAttributeReduction":
            #     red, score, runTime = reductionUseWeightedNeighborhood(data, radius, "POS", "PRE", X, Y) # 对应算法3 基于属性权重的的属性约简
            # elif algorithmName == "VariableRadiusNeighborhoodRoughSet":
            #     red, score, runTime = reductionUseVariableRadiusNeighborhoodRoughSet(X, Y, radius)  # 对应算法4 基于变邻域半径的属性约简
            # elif algorithmName == "RandomSamplingAttribureReduction":
            #     red, score, runTime = reductionUseRandomSampling(X, Y, radius, "PRE") # 对应算法5 基于随机样本分组的属性约简

            # 指定属性约简的算法
            if algorithmName == "algorithm0":
                red, score, runTime = reductionUseNeighborhoodRoughSet(X, Y, radius, "PRE") # 对应算法0 基于邻域粗糙集的属性约简
            elif algorithmName == "algorithm1":
                red, score, runTime = reductionUseAttributeGroup(data, radius, "POS", "PRE", X, Y) # 对应算法1 基于属性分组的属性约简
            elif algorithmName == "algorithm2_1":
                red, score, runTime = reductionUseDisSimilarity(data, radius, "POS", "PRE", X, Y) # 对应算法2-1 基于差异度的属性约简
            elif algorithmName == "algorithm2_2":
                red, score, runTime = reductionUseSimilarity(data, radius, "POS", "PRE", X, Y) # 对应算法2-2 基于相似度的属性约简
            elif algorithmName == "algorithm3":
                red, score, runTime = reductionUseWeightedNeighborhood(data, radius, "POS", "PRE", X, Y) # 对应算法3 基于属性权重的的属性约简
            elif algorithmName == "algorithm4":
                red, score, runTime = reductionUseVariableRadiusNeighborhoodRoughSet(X, Y, radius)  # 对应算法4 基于变邻域半径的属性约简
            elif algorithmName == "algorithm5":
                red, score, runTime = reductionUseRandomSampling(X, Y, radius, "PRE") # 对应算法5 基于随机样本分组的属性约简

            knnScore, svmScore, cartScore = getClassificationAccuracy(oriX, oriY, red)
            nowTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            resList = [nowTime, algorithmName, category, dataName, radius,
                        red, attrNum, len(red), score, runTime,
                        knnScore, svmScore, cartScore,
                        sum(knnScore)/len(knnScore), sum(svmScore)/len(svmScore), sum(cartScore)/len(cartScore)]
            resList = [str(e) for e in resList]

            print(resList)  # 展示运行进度

            resStorePath = os.path.join("../Res", "compareAlgorithms", algorithmName, category)
            if not os.path.exists(resStorePath):  # 检查目录是否存在如果不存在则创建目录
                os.makedirs(resStorePath)

            fileName = os.path.join(resStorePath, "{}.csv".format(dataName))
            if not os.path.exists(fileName):
                with open(fileName, "w", encoding="utf-8") as f:
                    f.write("结果生成时间|算法名称|数据集种类|数据集名称|半径|约简结果|原始数据属性长度|约简之后属性长度|属性集得分|运行时间|5折KNN准确率|5折SVM准确率|5折CART准确率|KNN平均准确率|SVM平均准确率|CART平均准确率\n")
                    f.write("|".join(resList) + "\n")
            else:
                with open(fileName, "a", encoding="utf-8") as f:  # 默认会进行追加
                    f.write("|".join(resList) + "\n")


# 对比实验进度



# storeDirPath = os.path.join(os.path.abspath("../Res"), "exp1", "compareAlgorithms", "compareAlgorithm4")
#
# # 检查特定的文件是否存在 如果不存在则创建文件 如果存在则往文件中追加写结果
# if not os.path.exists(storeDirPath):  # os模块判断并创建
#     os.makedirs(storeDirPath)
#
# for dataName in dataNames:
#     # 根据数据名读取数据 并对x进行归一化处理
#     path = '../DataSet_TEST/{}.csv'.format(dataName)
#     data = np.loadtxt(path, delimiter=",", skiprows=1)
#     sampelNum, attrNum = data.shape
#
#     X = data[:, :-1]
#     X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
#     Y = data[:, -1]
#
#
#     for radius in expRadius:
#         # 需要进行记录的信息
#         # 结果生成时间, 算法名称, 数据集名称, 约简结果, 依赖度, 约简时间
#         # nowTime, algorithmName, dataName, red, score, runTime
#         nowTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#
#         # 指定属性约简的算法
#         # red, score, runTime = reductionUseNeighborhoodRoughSet(X, Y, radius, "PRE") # 对应算法0 基于邻域粗糙集的属性约简
#         # red, score, runTime = reductionUseAttributeGroup(dataName, radius, "POS", "PRE", X, Y) # 对应算法1 基于属性分组的属性约简
#         # red, score, runTime = reductionUseDisSimilarity(dataName, radius, "POS", "PRE", X, Y) # 对应算法2-1 基于差异度的属性约简
#         # red, score, runTime = reductionUseSimilarity(dataName, radius, "POS", "PRE", X, Y) # 对应算法2-2 基于相似度的属性约简
#         # red, score, runTime = reductionUseWeightedNeighborhood(dataName, radius, "POS", "PRE", X, Y) # 对应算法3 基于属性权重的的属性约简
#         red, score, runTime = reductionUseVariableRadiusNeighborhoodRoughSet(X, Y, radius) # 对应算法4 基于变邻域半径的属性约简
#         # red, score, runTime = reductionUseRandomSampling(X, Y, radius, "PRE") # 对应算法5 基于随机样本分组的属性约简
#
#         resList = [nowTime, algorithmName, dataName, radius, red, score, runTime]
#         resList = [str(e) for e in resList]
#         print(resList)
#
#         fileName = os.path.join(storeDirPath, dataName + ".csv")
#         if not os.path.exists(fileName):
#             with open(fileName, "w", encoding="utf-8")as f:
#                 f.write("结果生成时间|算法名称|数据集名称|半径|约简结果|依赖度|约简时间\n")
#                 f.write("|".join(resList)+"\n")
#         else:
#             with open(fileName, "a", encoding="utf-8")as f: # 默认会进行追加
#                 f.write("|".join(resList)+"\n")


# res = reductionUseAttributeGroup("plrx", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 1
# print(res)

# res = reductionUseWeightedNeighborhood("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 3
# print(res)

# res = reductionUseDisSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 2-1
# print(res)

# res = reductionUseSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 2-2
# print(res)
