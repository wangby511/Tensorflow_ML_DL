from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import numpy as np

# 2019 MAY 6TH

def transfer(y, c):
    # make multi-class classification to one-verus-allOthers (binary classification)
    for i in range(len(y)):
        if y[i] == c:
            y[i] = 0
        else:
            y[i] = 1
    print(y.shape)
    return y

def testIris():
    #load data
    iris = load_iris()

    # 我们以sklearn中iris数据作为训练集，iris属性特征包括花萼长度、花萼宽度、花瓣长度、花瓣宽度
    # 类别共三类，分别为Setosa、Versicolour、Virginca

    X = iris.data
    y = iris.target
    print("X = ", X, " , y = ", y)
    clf = tree.DecisionTreeClassifier()
    print("original      y labels =",y)

    # change this number to 0,1 or 2
    # make multiclass classification to binary classification
    ### IMORTANT !!!
    # y = transfer(y, 0)


    print("after convert y labels =",y)
    clf = clf.fit(X, y)


    #export the decision tree
    #export_graphviz support a variety of aesthetic options
    dot_data = tree.export_graphviz(clf,
                                  out_file = None,
                                  feature_names = iris.feature_names,
                                  class_names = iris.target_names,
                                  filled = True,
                                  rounded = True,
                                  special_characters = True)

    graph = graphviz.Source(dot_data)
    graph.view()

def testMyOwnData():
    # data:
    X = np.array([i for i in range(1, 11)]).reshape(-1,1)
    split = 3.5
    y = [0 if x < split else 1 for x in X]
    y = [2, 2, 2, 1, 1, 1, 1, 1, 1, 0]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    dot_data = tree.export_graphviz(clf,
                                    out_file = None,
                                    filled = True,
                                    rounded = True,
                                    special_characters = True)
    graph = graphviz.Source(dot_data)
    graph.view()

# testIris()
testMyOwnData()