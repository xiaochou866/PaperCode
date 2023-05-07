import random

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
'''
    通过对原始数据集进行处理生成5折数据 以及 添加了噪声的数据
'''

def generateFoldXindex(sampleNum:int, foldNum:int):
    '''
    :param sampleNum: 该数据集的样本数量
    :param foldNum: 进行数据集划分的折数
    :return: 各个不同论域
    '''
    trainIndexArr = []
    kf = KFold(n_splits=foldNum, shuffle=False)
    for train_index, test_index in kf.split(list(range(sampleNum))):  # 调用split方法切分数据
        print(test_index)
        trainIndexArr.append(train_index)
    return trainIndexArr

def getAllPathUnderDir(dir: str):
    relPath = os.path.relpath(dir)
    # absPath = os.path.abspath(dir)
    # print("相对路径为{}, 绝对路径为{}".format(relPath, absPath))

    files = os.listdir(relPath)
    paths = [os.path.join(relPath, file) for file in files]
    return files, paths

def generateNoiseData(paths:list[str], files:list[str], noiseRates:list[int]):
    n = len(noiseRates)
    # 先生成用于存放加了多少噪声的数据目录
    for i in range(n):
        foldDir = "{}noise".format(noiseRates[i])
        if not os.path.isdir(foldDir):
            os.makedirs(foldDir)

    # 读取每一个路径下的数据
    for i in range(len(paths)):
        df = pd.read_csv(paths[i])
        sampleNum, attrNum = df.shape
        decValue = list(df.loc[:,"Class"].unique())

        for j in range(n):
            df = pd.read_csv(paths[i])
            dest = os.path.join("{}noise".format(noiseRates[j]), files[i])
            changeSampleNum = round(sampleNum*noiseRates[j]/100)
            changeSampelIdxs = random.sample(list(range(sampleNum)), changeSampleNum)

            for sampleIdx in changeSampelIdxs:
                oriDec = df.iloc[sampleIdx, -1]
                while True: # 将原有的决策属性值随机替换成另外一个属性值
                    changeDec = random.sample(decValue, 1)[0]
                    if changeDec!=oriDec:
                        print(sampleIdx)
                        df.iloc[sampleIdx, -1] = changeDec
                        break
            df.to_csv(dest, index=False)



def generateNFoldData(paths:list[str], files:list[str], N:int):
    # 先生成用于存放几折数据的目录
    for i in range(N):
        foldDir = "{}fold".format(i+1)
        if not os.path.isdir(foldDir):
            os.makedirs(foldDir)

    # 读取每一个路径下的数据
    for i in range(len(paths)):
        df = pd.read_csv(paths[i])
        sampleNum, attrNum = df.shape
        trainIndexArr = generateFoldXindex(sampleNum, N)
        for j in range(N):
            dest = os.path.join("{}fold".format(j+1), files[i])
            df.iloc[trainIndexArr[j],:].to_csv(dest, index=False)

if __name__ == "__main__":
    files, paths = getAllPathUnderDir("./ori")
    # generateNFoldData(paths, files, 5)
    generateNoiseData(paths, files, [10, 20, 30])

# data = np.loadtxt("./ori/iris.csv", skiprows=1, delimiter=",")
# print(data)

# df = pd.read_csv("./ori/iris.csv")
# df.iloc[0:10,:].to_csv("./iris.csv", index=False)
