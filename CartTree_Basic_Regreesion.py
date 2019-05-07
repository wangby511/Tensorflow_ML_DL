import numpy as np
inf = 1e9

# CART回归树预测回归连续型数据，假设X与Y分别是输入和输出变量，并且Y是连续变量。
# 在训练数据集所在的输入空间中，递归的将每个区域划分为两个子区域并决定每个子区域上的输出值，构建二叉决策树。
# 选择最优切分变量j与切分点s：遍历变量j，对规定的切分变量j扫描切分点s，选择使下式得到最小值时的(j,s)对。其中Rm是被划分的输入空间，cm是空间Rm对应的固定输出值。
# 用选定的(j,s)对，划分区域并决定相应的输出值。
# 继续对两个子区域调用上述步骤，将输入空间划分为M个区域R1,R2,…,Rm，生成决策树。
# https://zhuanlan.zhihu.com/p/36108972

def getLeftRight(X, y, split):
    n1 = 0
    n2 = 0
    c1 = 0
    c2 = 0
    for i in range(len(X)):
        x = X[i]
        if x < split:
            n1 = n1 + 1
            c1 = c1 + y[i]
        else:
            n2 = n2 + 1
            c2 = c2 + y[i]
    c1 = c1/n1
    c2 = c2/n2

    LeftMin = 0
    RightMin = 0
    for i in range(len(X)):
        x = X[i]
        if x < split:
            LeftMin += (y[i] - c1) * (y[i] - c1)
        else:
            RightMin += (y[i] - c2) * (y[i] - c2)
    return c1, c2, LeftMin, RightMin

def ContinuousTreeSplitTest(X, y):
    totalMax = inf
    bestSplit = -1
    C1 = -1
    C2 = -1
    for s in range(9):
        split = s + 1.5
        c1, c2, LeftMin, RightMin = getLeftRight(X, y, split)
        # print("split =", split)
        # print("leftMin = ",LeftMin)
        # print("RightMin = ",RightMin)
        if LeftMin + RightMin < totalMax:
            totalMax = LeftMin + RightMin
            bestSplit = split
            C1 = c1
            C2 = c2
    print("\nTree:")
    print("T(x) =",C1," if x <  ",bestSplit)
    print("T(x) =",C2," if x >= ",bestSplit)
    return bestSplit, C1, C2

def mainTest():
    # only two levels:

    # data:
    X = [i for i in range(1, 11)]
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9, 9.05]
    N = len(X)

    # y = [-0.68, -0.54, -0.33, 0.16, 0.56, 0.81, -0.01, -0.21, 0.09, 0.14]

    y_predict = [0 for i in range(N)]
    y_delta = y
    for iter in range(6):
        # 一共学习六轮


        # calculate for the split tree:
        split, C1, C2 = ContinuousTreeSplitTest(X, y_delta)
        y_current_classify = [C1 if x < split else C2 for x in X]
        y_predict = [y_current_classify[i] + y_predict[i] for i in range(N)]
        print("C1 =",C1,",C2 =",C2)
        print("y_predict =",y_predict)


        # 平方误差
        error = sum([(y[i] - y_predict[i]) * (y[i] - y_predict[i]) for i in range(N)])
        print("iteration:",iter,",error =", error)

        # 残差
        # 作为下一轮的学习的y
        y_delta = [y[i] - y_predict[i] for i in range(N)]

mainTest()