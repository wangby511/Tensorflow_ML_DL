import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

iris = datasets.load_iris()
#iris.data 150 x 4
#iris.target 150,
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

#split into training dataset and evalutaion datasets.
train_indices = np.random.choice(len(x_vals),
                                 round(len(x_vals) * 0.9),
                                 replace = False)

test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices] # 120 x 2
x_vals_test = x_vals[test_indices] # 30 x 2
y_vals_train = y_vals[train_indices] # 120,
y_vals_test = y_vals[test_indices] # 30,

batch_size = 110
# Initialize placeholders
x_data = tf.placeholder(shape = [None, 2], dtype = tf.float32)
y_target = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Create variables for SVM
w = tf.Variable(tf.random_normal(shape = [2, 1]))
b = tf.Variable(tf.random_normal(shape = [1, 1]))

# Declare model operations
model_output = tf.add(tf.matmul(x_data, w), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(w))

# Declare loss function
# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# L2 regularization parameter, alpha

alpha = tf.constant([0.01])

# Margin term in loss
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))

# Put terms together
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

# Declare optimizer
my_opt = tf.train.AdamOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

index = []
loss_vec = []
train_accuracy = []
test_accuracy = []
max_iterations = 4000
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

    # if (i + 1) % 5 == 0:
    #     print('Step #{} A = {}, b = {}'.format(
    #         str(i+1),
    #         str(sess.run(w)),
    #         str(sess.run(b))
    #     ))
    #     print('Loss = ' + str(temp_loss))

# Extract coefficients
[[a1], [a2]] = sess.run(w)
[[b]] = sess.run(b)
#a1 * x1 + a2 * x2 + b = 0
#x1 = -1/a1*(b + a1 * x1)
slope = -a2/a1
y_intercept = -b/a1

# Extract x1 and x2 vals
x1_vals = [d[1] for d in x_vals]

# Get best fit line
best_fit = []
for i in x1_vals:
    best_fit.append(slope*i + y_intercept)

print("best_fit = ",best_fit)

# Separate I. setosa
setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

# matplotlib inline
# Plot data and line
plt.plot(setosa_x, setosa_y, 'o', label = 'I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label = 'Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label = 'Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc = 'lower right')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

# print("loss_vec = ",loss_vec)
plt.plot(index,train_accuracy)
plt.plot(index,loss_vec)
# print("train_accuracy = ",train_accuracy)
plt.show()
