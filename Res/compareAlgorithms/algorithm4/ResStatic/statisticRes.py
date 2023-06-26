import os
import pandas as pd


# 设置
# accPredAlgorithm = "KNN"
accPredAlgorithm = "SVM"
# accPredAlgorithm = "CART"

dataCategory = ["ori", "1fold", "2fold", "3fold", "4fold", "5fold", "10noise", "20noise", "30noise"]
category = dataCategory[0]

modes = ["optimal", "medium", "avg"]
mode = modes[0]

# 根据配置项指定一些变量的值 用于后续结果统计
baseDir = "../{}/".format(category)
foldColName = "5折{}准确率".format(accPredAlgorithm)
avgColName = "{}平均准确率".format(accPredAlgorithm)
algorithmName = os.path.abspath('../').split('\\')[-1]


res = pd.DataFrame(columns=['数据集', '1-fold', '2-fold', '3-fold', '4-fold', '5-fold', avgColName])
files = os.listdir(baseDir)  # 获取结果目录下的所有数据

for file in files:
    if not file.endswith("csv"): continue

    # 按实验结果产生的日期进行一个筛选
    df = pd.read_csv(baseDir+file, delimiter='|')
    recordNum, _ = df.shape

    # region 这里要根据一定规则 选出 一个 数据出来
    df.sort_values(by=avgColName, inplace=True, ascending=False, ignore_index=True) # 选择一个KNN平均准确率最大的一条实验记录
    if mode == "optimal":
        dataSetName = df.loc[0, "数据集名称"]
        foldAcc = df.loc[0, foldColName][1:-1].split()
        avgAcc = df.loc[0, avgColName]
    elif mode == "medium":
        dataSetName = df.loc[recordNum//2, "数据集名称"]
        foldAcc = df.loc[recordNum//2, foldColName][1:-1].split()
        avgAcc = df.loc[recordNum//2, avgColName]
    elif mode == "avg":
        dataSetName = df.loc[0, "数据集名称"]
        foldAcc =[0]*5
        for i in range(recordNum):
            tmp = df.loc[i, foldColName][1:-1].split()
            for j in range(5):
                foldAcc[j] += float(tmp[j])
        foldAcc = [e/recordNum for e in foldAcc]
        avgAcc = df[avgColName].mean()



    # endregion

    # 往dataFrame中添加一条数据
    res = res.append({'数据集': dataSetName,
                    '1-fold': foldAcc[0], '2-fold': foldAcc[1], '3-fold': foldAcc[2],
                    '4-fold': foldAcc[3], '5-fold': foldAcc[4],
                    avgColName: avgAcc}, ignore_index=True)

res.to_excel("{}-{}-{}-{}.xlsx".format(algorithmName, category, mode, accPredAlgorithm))