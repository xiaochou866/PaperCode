import os
import numpy as np
import pandas as pd

# os.listdir() 方法获取文件夹名字，返回数组
# def getAllFiles(targetDir):
#     listFiles = os.listdir(targetDir)
#     return listFiles
#
# files = getAllFiles(".")
#
# for file in files:
#     print(file)

# 处理 BreastCancer数据集 将字符串类型转化为整型 并使用前向填充缺失值的方式 最后去掉第一列id
# path = '../DataSet_TEST/{}.csv'.format("BreastCancer")
# df = pd.read_csv(path)
# df.astype('int', errors='ignore')
# df.fillna(method='ffill')
# df.iloc[:, 1:].to_csv("BreastCancer1.csv", index=False)

# 处理College数据集 将第一列的值进行转换 Yes->0 No->1 并将第一列移动到最后一列 采用先pop 再insert的方法
# path = '../DataSet_TEST/{}.csv'.format("College")
# df = pd.read_csv(path)
# df.iloc[:, 0] = df.iloc[:, 0].apply(lambda s: 0 if s == "Yes" else 1)
# temp_column = df.pop('Private')
# df.insert(df.shape[1], 'Private', temp_column)
# print(df)
# df.to_csv("College.csv", index=False)


# 处理diamonds_filter数据集将最后一列进行转换
# def changeColumn(s):
#     if s == 'Fair':
#         return 0
#     elif s == 'Good':
#         return 1
#     elif s == 'Ideal':
#         return 2
#     elif s == 'Premium':
#         return 3
#
#
# path = '../DataSet_TEST/{}.csv'.format("diamonds_filter")
# df = pd.read_csv(path)
# columnNum = df.shape[1]
# df.iloc[:, columnNum - 1] = df.iloc[:, columnNum - 1].apply(changeColumn)
# df.to_csv("diamonds_filter.csv", index=False)

# # 根据决策属性对数据集进行重排
# dir = "30noise"
# files = os.listdir(dir)
# for file in files:
#     path = os.path.join(dir, file)
#     df = pd.read_csv(path)
#     df = df.sort_values(by=['Class'], ascending=True, ignore_index=True);
#     df.to_csv(path, index=False)

# 生成一个统一的表头
# sampleNum, attrNum = df.shape
# c_list = ["V{}".format(i) for i in range(1, attrNum)]
# c_list.append("Class")
# df.columns = c_list


# 将决策列转化为整数
# df["Class"] = df["Class"].astype(int)

# 整理各个数据集的情况生成excel文件
df = pd.DataFrame(columns=["数据集", "样本数", "属性数", "类别数"])
Files = os.listdir("./ori")
for file in Files:
    if file.endswith(".csv"):
        data = pd.read_csv(os.path.join("ori", file))
        sampleNum, attrNum = data.shape
        newRow = [file.split('.')[0], sampleNum, attrNum, data['Class'].max()]
        df.loc[len(df)] = newRow
df = df.sort_values(by=['样本数'], ascending=True, ignore_index=True);
print(df.iloc[:,0].tolist())

# df = df.sort_values(by=['属性数'], ascending=True, ignore_index=True);
# df.to_excel("数据详情-基于属性数升序.xlsx")

# df = df.sort_values(by=['类别数'], ascending=True, ignore_index=True);
# df.to_excel("数据详情-基于类别数升序.xlsx")

# if __name__ == "__main__":
#     path = '../DataSet_TEST/{}.csv'.format("BreastCancer")
#     df = pd.read_csv(path)
#     df = df.fillna(method='ffill')
#     print(df.iloc[10:30,:])
#     df.to_csv(path, index=False)












