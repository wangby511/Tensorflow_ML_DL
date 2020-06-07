import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
# Reference : https://www.jianshu.com/p/33b3da67635a

# read data
# df = pd.read_csv('boston.csv')
y_data = load_boston().target
x_data = load_boston().data
# print(x_data.shape)
# print(df.describe())
# print(df)

# df = df.values
# print(df)
# df = np.array(df,np.float32) #convert to numpy
#
feature_number = x_data.shape[1] - 1
x_data = x_data[:, :feature_number]
# x_data = df[:, :feature_number]
# y_data = df[:, feature_number]

# normalization
for i in range(x_data.shape[1]):
    x_data[:,i] /= (x_data[:,i].max() - x_data[:,i].min())

# print('x_data shape = ', x_data.shape) # (506, 13)
# print('y_data shape = ', y_data.shape) # (506,)

x = tf.compat.v1.placeholder(tf.float32, [None, feature_number], name = 'x')
y = tf.compat.v1.placeholder(tf.float32, [None, ], name = 'y')

w = tf.Variable(tf.random.normal([feature_number, 1], stddev = 0.1),name = 'w')
b = tf.Variable(1.0, name='b')
predict = tf.matmul(x, w) + b

train_epochs = 500 # epoch number
learning_rate = 0.01 # learning rate

loss_function = tf.reduce_mean(tf.pow(y - predict, 2)) #MSE

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    loss_sum = 0
    loss_ave_rec = []

    for epoch in range(train_epochs):
        batch = np.random.randint(0, x_data.shape[0], 20)

        _, loss = sess.run([optimizer, loss_function], feed_dict={x: x_data[batch], y: y_data[batch]})

        loss_sum += loss
        loss_ave_rec.append(loss_sum / (epoch + 1))
        # print(loss)

        if epoch % 50 == 0:
            _, final_loss = sess.run([optimizer, loss_function], feed_dict={x: x_data, y: y_data})
            print("epoch:", epoch, ",loss =", final_loss)

    plt.figure()
    plt.plot(loss_ave_rec)
    # plt.show()
    # _, predicts = sess.run([optimizer, predict], feed_dict={x: x_data, y: y_data})
    # print("epoch:", epoch, ",predicts =", predicts)