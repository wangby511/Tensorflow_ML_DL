import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb
from sklearn.datasets import load_iris
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from graphviz import Digraph
import pydot

# http://github.com/wanglei5205
### load datasets
digits = datasets.load_digits()
digits = load_iris()
### data analysis
# print(digits.data.shape)
# print(digits.target.shape)
# (1797, 64)
# (1797,)

### data split
x_train,x_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size = 0.3,
                                                 random_state = 33)

print(x_train.shape)
print("x_train =",x_train)
print(y_train.shape)
print("y_train =",y_train)
classesNumber = len(set(y_train))
# print(len(set(y_train))) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
print(x_test.shape)
print(y_test.shape)



### fit model for train data
# 载入数据：load_digits()
# 数据拆分：train_test_split()
# 建立模型：XGBClassifier()
# 模型训练：fit()
# 模型预测：predict()
# 性能度量：accuracy_score()
# 特征重要性：plot_importance()

max_Iterations = 1

model = XGBClassifier(learning_rate=0.1,
                      silent = 0,
                      n_estimators = max_Iterations,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      # subsample=0.8,             # 随机选择80%样本建立决策树
                      # colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      subsample = 1,
                      colsample_btree = 1,
                      objective='multi:softmax', # 指定损失函数
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27            # 随机数
                      )
model.fit(x_train,
          y_train,
          eval_set = [(x_test,y_test)],
          eval_metric = "mlogloss",
          # early_stopping_rounds = 10,
          verbose = True)

## plot feature importance
# fig,ax = plt.subplots(figsize = (15,15))
# plot_importance(model,
#                 height = 0.5,
#                 ax = ax,
#                 max_num_features = 64)
# plt.show()
# for i in range(classesNumber):
#     plot_tree(model, num_trees = i)
#     plt.show()
plot_tree(model, num_trees = 2)
plt.show()
### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate
accuracy = accuracy_score(y_test, y_pred)
print("accuarcy: %.2f%%" % (accuracy * 100.0))

### test for iter 1 tree 0:
print("\n\ntest for iter 1 tree 0:")
list1 = []
list2 = []
for i in range(len(x_train)):
    if x_train[i][2] > 2.45:
        list1.append(y_train[i])
    else:
        list2.append(y_train[i])

print("list1 = ",list1)
print("list2 = ",list2)

### test for iter 1 tree 1:
print("\n\ntest for iter 1 tree 1:")
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
for i in range(len(x_train)):
    sample = x_train[i]
    if sample[2] < 2.45:
        list1.append(y_train[i])
    else:
        if sample[3] < 1.75:
            if sample[2] < 5.05:
                if sample[0] < 5.05:
                    list2.append(y_train[i])
                else:
                    list3.append(y_train[i])
            else:
                list4.append(y_train[i])
        else:
            if sample[2] < 4.9499:
                list5.append(y_train[i])
            else:
                list6.append(y_train[i])
        list2.append(y_train[i])

print("list1 = ",list1)
print("list2 = ",list2)
print("list3 = ",list3)
print("list4 = ",list4)
print("list5 = ",list5)
print("list6 = ",list6)

### test for iter 1 tree 2:
print("\n\ntest for iter 1 tree 2:")
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
for i in range(len(x_train)):
    sample = x_train[i]
    if sample[3] < 1.65:
        if sample[2] < 5:
            list1.append(y_train[i])
        else:
            list2.append(y_train[i])
    else:
        if sample[2] < 5.05:
            if sample[1] < 2.85:
                list3.append(y_train[i])
            else:
                list4.append(y_train[i])
        else:
            list5.append(y_train[i])

print("list1 = ",list1)
print("list2 = ",list2)
print("list3 = ",list3)
print("list4 = ",list4)
print("list5 = ",list5)
# print("list6 = ",list6)




