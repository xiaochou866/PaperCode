import os
import pandas as pd

res = pd.DataFrame(columns=['数据集', '1-fold', '2-fold', '3-fold', '4-fold', '5-fold', 'KNN平均准确率'])
files = os.listdir('.')  # 获取结果目录下的所有数据
for file in files:
    if file.endswith("py"): continue

    df = pd.read_csv(file, delimiter='|')
    df.sort_values(by="KNN平均准确率", inplace=True, ascending=False, ignore_index=True) # 选择一个KNN平均准确率最大的一条实验记录
    dataSetName = df.loc[0, "数据集名称"]
    foldAcc = df.loc[0, "5折KNN准确率"][1:-1].split()
    avgAcc = df.loc[0, "KNN平均准确率"]
    # print(dataSetName)
    # print(foldAcc)
    # print(avgAcc)
    res = res.append({'数据集': dataSetName,
                    '1-fold': foldAcc[0], '2-fold': foldAcc[1], '3-fold': foldAcc[2],
                    '4-fold': foldAcc[3], '5-fold': foldAcc[4],
                    'KNN平均准确率': avgAcc}, ignore_index=True)

res.to_excel("myAlgorithm6-KNN.xlsx")