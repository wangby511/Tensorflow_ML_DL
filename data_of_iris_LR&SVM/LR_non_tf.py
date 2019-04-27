import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

iris = datasets.load_iris()
#iris.data 150 x 4
#iris.target 150,
x_vals = np.array(iris.data)
y_vals = np.array([0 if y == 0 else 1 for y in iris.target])
D = x_vals.shape[1]
N = x_vals.shape[0]
x_vals_train = x_vals # 150 x 4
y_vals_train = y_vals # 150,

index = []
loss_vec = []
accuracy_vec = []

# Initialize variables
w = np.zeros((4,1))
b = 0

# RUN
max_iterations = 50

batch_size = np.minimum(N,10)

for i in range(max_iterations):

    #1.random choose some data as training epoch

    rand_index = np.random.choice(len(x_vals_train), size = batch_size)

    x_data = x_vals_train[rand_index]

    y_label = np.transpose([y_vals_train[rand_index]])

    y_predict = sigmoid(np.dot(x_data,w) + b)

    #2.calculate accuracy

    y_whole_predict = sigmoid(np.dot(x_vals,w) + b)

    y_whole_predict_labels = [1 if x >= 0.5 else 0 for x in y_whole_predict]

    accu = 0

    for j in range(N):
        if y_whole_predict_labels[j] == y_vals[j]:
            accu = accu + 1

    train_accuracy = accu * 1.0 / N

    #3.update weight

    ones = np.ones((batch_size, 1))

    log_term1 = np.maximum(y_predict,1e-2)

    log_term2 = np.maximum(ones - y_predict,1e-2)

    loss = -(np.multiply(y_label,np.log(log_term1)) + np.multiply((ones - y_label),np.log(log_term2)))

    if i % 10 == 0:
        print("loss = ", np.sum(loss),", train_accuracy = ",train_accuracy)

    loss_vec.append(np.sum(loss))

    accuracy_vec.append(train_accuracy)

    # if train_accuracy < 1.0:
    #     print("iter = ",i)

    lr = 0.1

    _lambda = 0.05

    w -= lr * (np.sum(np.multiply(y_predict - y_label,x_data),axis = 0).reshape(4,1) + _lambda * w)

    b -= lr * (np.sum(y_predict - y_label) + _lambda * b)

    index.append(i)

    # if np.sum(loss) > 10.0:
    #     print(y_predict.reshape(batch_size))
    #     print(y_label.reshape(batch_size))


# Extract coefficients
print("\n")
print("w = ",w)
print("b = ",b)

loss_vec_array = np.array(loss_vec)
loss_vec_array = loss_vec_array / np.max(loss_vec_array)

plt.plot(index,loss_vec_array)
plt.plot(index,accuracy_vec)
plt.xlabel('Loss & Accuracy')
plt.show()