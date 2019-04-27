import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
#iris.data 150 x 4
#iris.target 150,
x_vals = np.array(iris.data)
y_vals = np.array(iris.target)

x_vals_train = x_vals # 150 x 4
y_vals_train = y_vals # 150,
print(x_vals_train.shape)

selectedD1 = 1
selectedD2 = 2
setosa_x_class1 = [d[selectedD1] for i, d in enumerate(x_vals) if y_vals[i] == 0]
setosa_y_class1 = [d[selectedD2] for i, d in enumerate(x_vals) if y_vals[i] == 0]

setosa_x_class2 = [d[selectedD1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y_class2 = [d[selectedD2] for i, d in enumerate(x_vals) if y_vals[i] == 1]

setosa_x_class3 = [d[selectedD1] for i, d in enumerate(x_vals) if y_vals[i] == 2]
setosa_y_class3 = [d[selectedD2] for i, d in enumerate(x_vals) if y_vals[i] == 2]

# matplotlib inline
# Plot data and line
plt.plot(setosa_x_class1, setosa_y_class1, 'o', label = 'Class_label_0')
plt.plot(setosa_x_class2, setosa_y_class2, 'x', label = 'Class_label_1')
plt.plot(setosa_x_class3, setosa_y_class3, 'D', label = 'Class_label_2')

plt.legend(loc = 'lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()