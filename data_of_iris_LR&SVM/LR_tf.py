import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

iris = datasets.load_iris()
#iris.data 150 x 4
#iris.target 150,
x_vals = np.array(iris.data)
y_vals = np.array([0 if y == 0 else 1 for y in iris.target])
D = x_vals.shape[1]
x_vals_train = x_vals # 150 x 4
y_vals_train = y_vals # 150,

batch_size = 50
# Initialize placeholders
x_data = tf.placeholder(shape = [None, D], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for LR
w = tf.Variable(tf.random_normal(shape = [D, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, w), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = model_output, labels = y_target))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

## RUN
index = []
loss_vec = []
train_accuracy = []
max_iterations = 400

for i in range(max_iterations):
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    # print(rand_index.shape)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    #
    temp_loss = sess.run(loss, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    #
    train_acc_temp = sess.run(accuracy, feed_dict = {
        x_data: x_vals_train,
        y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    #
    # test_acc_temp = sess.run(accuracy, feed_dict={
    #     x_data: x_vals_test,
    #     y_target: np.transpose([y_vals_test])})
    # test_accuracy.append(test_acc_temp)
    index.append(i)

    if (i + 1) % 20 == 0:
        print('iter = ',i + 1,', loss = ' + str(temp_loss) + ', accuracy = ',train_acc_temp)

# Extract coefficients
w = np.array(sess.run(w)).reshape(D,)
b = np.array(sess.run(b)).reshape(1,)

print("\n")
print("w = ",w)
print("b = ",b)
# w =  [ 0.3690526  -1.3538098   0.35085484  0.6481333 ]
# b =  [1.0982139]
# w =  [ 0.60513103 -1.2382102   0.06141503  1.6762964 ]
# b =  [-0.5984204]
# w =  [-0.30551615  0.37247992  0.45643693  1.2109575 ]
# b =  [-1.295734]