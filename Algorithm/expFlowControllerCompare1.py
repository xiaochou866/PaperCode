import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import MinMaxScaler

# 将要进行实验的所有对比算法
from Algorithm.compareAlgorithm0 import reductionUseNeighborhoodRoughSet
from Algorithm.compareAlgorithm1 import reductionUseAttributeGroup
from Algorithm.compareAlgorithm2_1 import reductionUseDisSimilarity
from Algorithm.compareAlgorithm2_2 import reductionUseSimilarity
from Algorithm.compareAlgorithm3 import reductionUseWeightedNeighborhood
from Algorithm.compareAlgorithm4 import reductionUseVariableRadiusNeighborhoodRoughSet
from Algorithm.compareAlgorithm5 import reductionUseRandomSampling

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
    # 设置将要进行实验的半径
    expRadius = generateRadiusArr(0.02, 0.42, 0.02) # 设置实验半径为 0.02, 0.04, ... ,0.4 这20个半径

    # 设置将要进行实验的所有数据集
    # dataNames = ['fertility_Diagnosis', 'BreastTissue', 'Iris']  # 预先测试的三个小数据集
    dataNames = ['fertility_Diagnosis', 'BreastTissue', 'Iris', 'wine', 'plrx', 'GlaucomaM', 'Sonar', 'seeds', 'Glass', 'accent',
        'PimaIndiansDiabetes', 'Ionosphere', 'movement', 'vote', 'musk', 'wdbc', 'diamonds_filter', 'australian',
        'BreastCancer', 'diabetes', 'pima', 'College', 'Vehicle', 'german', 'data_banknote', 'waveform']
    # 运行到BreastCancer了 继续遇到数据集有缺失报错 修正数据集继续进行实验
    # dataNames = ['BreastCancer', 'diabetes', 'pima', 'College', 'Vehicle', 'german', 'data_banknote', 'waveform']


    # 设置将要进行实验的所有算法
    # algorithms = []
    # algorithmName = "NeighborhoodRoughSet" # 对应算法0
    # algorithmName = "AttributeGroupAttributeReduction" # 对应算法1
    # algorithmName = "DisSimilarityAttributeReduction" # 对应算法2-1
    # algorithmName = "SimilarityAttributeReduction" # 对应算法2-2
    # algorithmName = "WeightedAttributeReduction" # 对应算法3
    algorithmName = "VariableRadiusNeighborhoodRoughSet" # 对应算法4
    # algorithmName = "RandomSamplingAttribureReduction" # 对应算法5


    # 优先级 算法>数据集>半径
    storeDirPath = os.path.join(os.path.abspath("../Res"), "exp1", "compareAlgorithms", "compareAlgorithm4")

    # 检查特定的文件是否存在 如果不存在则创建文件 如果存在则往文件中追加写结果
    if not os.path.exists(storeDirPath):  # os模块判断并创建
        os.makedirs(storeDirPath)

    for dataName in dataNames:
        # 根据数据名读取数据 并对x进行归一化处理
        path = '../DataSet_TEST/{}.csv'.format(dataName)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        sampelNum, attrNum = data.shape

        X = data[:, :-1]
        X = MinMaxScaler().fit_transform(X)  # 归一化取值均归为0-1之间
        Y = data[:, -1]


        for radius in expRadius:
            # 需要进行记录的信息
            # 结果生成时间, 算法名称, 数据集名称, 约简结果, 依赖度, 约简时间
            # nowTime, algorithmName, dataName, red, score, runTime
            nowTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 指定属性约简的算法
            # red, score, runTime = reductionUseNeighborhoodRoughSet(X, Y, radius, "PRE") # 对应算法0 基于邻域粗糙集的属性约简
            # red, score, runTime = reductionUseAttributeGroup(dataName, radius, "POS", "PRE", X, Y) # 对应算法1 基于属性分组的属性约简
            # red, score, runTime = reductionUseDisSimilarity(dataName, radius, "POS", "PRE", X, Y) # 对应算法2-1 基于差异度的属性约简
            # red, score, runTime = reductionUseSimilarity(dataName, radius, "POS", "PRE", X, Y) # 对应算法2-2 基于相似度的属性约简
            # red, score, runTime = reductionUseWeightedNeighborhood(dataName, radius, "POS", "PRE", X, Y) # 对应算法3 基于属性权重的的属性约简
            red, score, runTime = reductionUseVariableRadiusNeighborhoodRoughSet(X, Y, radius) # 对应算法4 基于变邻域半径的属性约简
            # red, score, runTime = reductionUseRandomSampling(X, Y, radius, "PRE") # 对应算法5 基于随机样本分组的属性约简

            resList = [nowTime, algorithmName, dataName, radius, red, score, runTime]
            resList = [str(e) for e in resList]
            print(resList)

            fileName = os.path.join(storeDirPath, dataName + ".csv")
            if not os.path.exists(fileName):
                with open(fileName, "w", encoding="utf-8")as f:
                    f.write("结果生成时间|算法名称|数据集名称|半径|约简结果|依赖度|约简时间\n")
                    f.write("|".join(resList)+"\n")
            else:
                with open(fileName, "a", encoding="utf-8")as f: # 默认会进行追加
                    f.write("|".join(resList)+"\n")


    # res = reductionUseAttributeGroup("plrx", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 1
    # print(res)

    # res = reductionUseWeightedNeighborhood("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 3
    # print(res)

    # res = reductionUseDisSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 2-1
    # print(res)

    # res = reductionUseSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]]) 2-2
    # print(res)
