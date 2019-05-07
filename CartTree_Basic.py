from sklearn.datasets import load_iris
from sklearn import tree

def transfer(y, c):
    # make multi-class classification to one-verus-allOthers (binary classification)
    for i in range(len(y)):
        if y[i] == c:
            y[i] = 0
        else:
            y[i] = 1
    print(y.shape)
    return y

#load data
iris = load_iris()

# 我们以sklearn中iris数据作为训练集，iris属性特征包括花萼长度、花萼宽度、花瓣长度、花瓣宽度
# 类别共三类，分别为Setosa、Versicolour、Virginca

X = iris.data
y = iris.target
# print("X = ", X, " , y = ", y)
clf = tree.DecisionTreeClassifier()
print("original       y labels =",y)

# change this number to 0,1 or 2
# make multiclass classification to binary classification
y = transfer(y, 0)


print("after transfer y labels =",y)
clf = clf.fit(X, y)


#export the decision tree
import graphviz
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
