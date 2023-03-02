import numpy as np
import datetime
import json
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from Algorithm.compareAlgorithm1 import reductionUseAttributeGroup
from Algorithm.compareAlgorithm3 import reductionUseWeightedNeighborhood
from Algorithm.compareAlgorithm2_1 import reductionUseDisSimilarity
from Algorithm.compareAlgorithm2_2 import reductionUseSimilarity

# https://huaweicloud.csdn.net/637f7c9edacf622b8df85fef.html?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-1-116241388-blog-120377395.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-1-116241388-blog-120377395.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2
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

# 用于生成五折交叉验证的x的索引
def generateFoldXindex(sampleNum:int, foldNum:int):
    '''
    :param sampleNum: 该数据集的样本数量
    :param foldNum: 进行数据集划分的折数
    :return: 各个不同论域
    '''
    trainIndexArr = []
    kf = KFold(n_splits=foldNum, shuffle=False)
    for train_index, test_index in kf.split(list(range(sampleNum))):  # 调用split方法切分数据
        trainIndexArr.append(train_index)
    return trainIndexArr



if __name__ == "__main__":
    # 设置将要进行实验的数据集
    # dataNames = ["iris", "wine", "Ionosphere", "Glass", "data_banknote", "Sonar", "fertility-Diagnosis", "accent", "plrx", "wdbc",  "movement", "BreastTissue"]
    # dataNames = ["iris", "wine", "Ionosphere", "Glass", "data_banknote", "Sonar", "fertility-Diagnosis", "accent", "plrx", "wdbc",  "movement", "BreastTissue"]
    # dataNames = ["iris", "wine", "Ionosphere", "Glass", "data_banknote"]
    # dataNames = ["iris", "wine", "Ionosphere"]
    dataNames = [ "Glass", "data_banknote", "Sonar"]
    # dataNames = [ "accent", "plrx", "wdbc",  "movement", "BreastTissue"]

    algorithmName = "compareAlgorithm3"

    # 优先级 算法>数据集>折>半径
    for dataName in dataNames:
        # 根据数据名读取数据 并对x进行归一化处理
        path = '../DataSet_TEST/{}.csv'.format(dataName)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        sampelNum = data.shape[0]
        x = data[:, :-1]
        x = MinMaxScaler().fit_transform(x)  # 归一化取值均归为0-1之间
        y = data[:, -1]
        y = y.astype(np.int)

        # 根据样本总数和折数 划分不同的论域
        foldNum = 5
        trainIndexArr = generateFoldXindex(sampelNum, foldNum)

        # 指定起始半径 终止半径 和 间隔 生成将要进行实验的半径
        radiusArr = generateRadiusArr(0.04, 0.4, 0.04)

        for i in range(foldNum):
            foldName = "第{}折".format(i+1)
            indexs = trainIndexArr[i]
            foldX = x[indexs,:]
            foldY = y[indexs]

            for radius in radiusArr:
                res = reductionUseWeightedNeighborhood(dataName, radius, "POS", "PRE", foldX, foldY)
                print(res)
                # 对结果进行记录
                # 算法 以文件夹名进行标识
                # 数据集名: dataName 以文件夹名进行标识
                # 折: i+1
                # 半径: radius
                # 约简: res[0]
                # 得分: res[1]
                # 时间: res[2]
                # 日期: 记录结果产生的时间
                resDict = dict()
                resDict["algorithm"] = algorithmName
                resDict["radius"] = radius
                resDict["dataSet"] = dataName
                resDict["fold"] = i+1
                resDict["red"] = list(res[0])
                resDict["score"] = res[1]
                resDict["runtime"] = res[2]
                resDict["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 检查特定的文件是否存在 如果不存在则创建文件 如果存在则往文件中追加写结果
                dirName = os.path.join(os.path.abspath("../Res"), "compareAlgorithm", algorithmName)
                if not os.path.exists(dirName):  # os模块判断并创建
                    os.makedirs(dirName)

                fileName = os.path.join(dirName, dataName + ".json")
                if not os.path.exists(fileName):
                    with open(fileName, "w", encoding="utf-8")as f:
                        jsonStr = {foldName:[]}
                        jsonStr[foldName].append(resDict)
                        json.dump(jsonStr, f, ensure_ascii=False)
                else:
                    with open(fileName, "r+", encoding="utf-8")as f: # 默认会进行追加
                        jsonStr = json.load(f)
                        if foldName not in jsonStr:
                            jsonStr[foldName] = []
                        jsonStr[foldName].append(resDict)

                        # 读完 调整位置再写
                        f.seek(0)
                        f.truncate()
                        json.dump(jsonStr, f, ensure_ascii=False)




    # res = reductionUseAttributeGroup("plrx", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]])
    # print(res)

    # res = reductionUseWeightedNeighborhood("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]])
    # print(res)

    # res = reductionUseDisSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]])
    # print(res)

    # res = reductionUseSimilarity("iris", 0.2, "POS", "PRE", x[trainIndexArr[0],:], y[trainIndexArr[0]])
    # print(res)
