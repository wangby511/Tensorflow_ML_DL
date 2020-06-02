import os

import numpy as np
import tensorflow as tf

# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

input_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])

# 或者随机一个矩阵
embedding = np.asarray([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]])

# 根据input_ids中的id，查找embedding中对应的元素
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))
# [[1.1 1.2 1.3]
#  [2.1 2.2 2.3]
#  [3.1 3.2 3.3]
#  [0.1 0.2 0.3]
#  [3.1 3.2 3.3]
#  [2.1 2.2 2.3]
#  [1.1 1.2 1.3]]