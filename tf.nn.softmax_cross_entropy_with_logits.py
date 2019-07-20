import tensorflow as tf
import numpy as np

# tf.nn.softmax_cross_entropy_with_logits 2019.7.20 AFTERNOON SUZALLO LIBRARY

# NN's output
logits = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
# step1:do softmax
y = tf.nn.softmax(logits)
# true label
y_ = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
# step2:do cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# do cross_entropy just one step
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))  # dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    sess_y = sess.run(y)
    sess_cross_entropy = sess.run(cross_entropy)
    sess_cross_entropy2 = sess.run(cross_entropy2)
    print("step1:softmax result=\n",sess_y)
    print("step2:cross_entropy result=\n",sess_cross_entropy)
    print("step3:softmax_cross_entropy_with_logits) result=\n",sess_cross_entropy2)

print (np.log(0.66524094) * 3)
print (np.exp(3)/(np.exp(1) + np.exp(2) + np.exp(3)))

Truth = np.array([0,0,1,0])
Pred_logits = np.array([3.5,2.1,7.89,4.4])

loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=Truth,logits=Pred_logits) #f1
loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Truth,logits=Pred_logits) #f2
loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(Truth),logits=Pred_logits) #f3
# tf.argmax(Truth) = 2

with tf.Session() as sess:
    print("loss1 =",sess.run(loss1))
    print("loss2 =",sess.run(loss2))
    print("loss3 =",sess.run(loss3))

# f1的要求labels的格式和logits类似，比如[0,0,1,0]。
# 而f3的要求labels是一个数值，这个数值记录着ground truth所在的索引。
# 以[0,0,1,0]为例，这里真值1的索引为2。所以f3要求labels的输入为数字2(tensor)。
# 一般可以用tf.argmax()来从[0,0,1,0]中取得真值的索引。


# f1和f2之间很像，实际上官方文档已经标记出f1已经是deprecated 状态，推荐使用f2。
# 两者唯一的区别在于f1在进行反向传播的时候，只对logits进行反向传播，labels保持不变。
# 而f2在进行反向传播的时候，同时对logits和labels都进行反向传播，如果将labels传入的tensor设置为stop_gradients，就和f1一样了。
# 那么问题来了，一般我们在进行监督学习的时候，labels都是标记好的真值，什么时候会需要改变label？f2存在的意义是什么？
# 实际上在应用中labels并不一定都是人工手动标注的，有的时候还可能是神经网络生成的，一个实际的例子就是对抗生成网络（GAN）。
# https://blog.csdn.net/tsyccnh/article/details/81069308
# https://blog.csdn.net/zj360202/article/details/78582895

"""
Back-propagation of softmax:
yi = exp(xi) / sigma(j = 1,...,K)exp(xj)

d(yi)/d(xj) = yi(1 - yj) when i == j
            = yi(0 - yj) when i != j


C = -sigma(i = 1,...,C)yi^ * log(yi) = - log(yi) i为真实的标签
d(C)/d(xi) = d(C)/d(yi) * d(yi)/d(xi) = - (1 / yi) * yi(1 - yi)  = yi - 1  when i == j

d(C)/d(xi) = d(C)/d(yi) * d(yi)/d(xi) = - (1 / yi) * yi(0 - yj)  = yj - 0  when i == j

so, d(C)/d(xk) = yk - yk^  (yk^ 为真实的标签)
"""