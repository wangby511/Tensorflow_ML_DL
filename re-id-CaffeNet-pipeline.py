import numpy as np
from AlexNet_tf import AlexNet
from CaffeNet_tf import CaffeNet
import tflearn.datasets.oxflower17 as oxflower17
import random

X,Y = oxflower17.load_data(one_hot = True,resize_pics = (227,227))

N = X.shape[0]
permu_index = np.random.permutation(N)
print(permu_index)
X = X[permu_index]
Y = Y[permu_index]
print("X.shape = ",X.shape) # (1360, 227, 227, 3)
print("Y.shape = ",Y.shape) # (1360, 17)
# print(Y)
# alexnet = AlexNet(
#     batch_size=128,
#     image_size=227,
#     class_number=17,
#     epoch=1
# )
#
# # print(label_train.shape)
# alexnet.fit(X,Y)

caffenet = CaffeNet(
    batch_size=128,
    image_size=227,
    class_number=17,
    epoch=1
)
caffenet.fit(X,Y)

# 2019-09-25 17:17:06.990496: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# epoch : 0  j = 0  loss = 3.9730442
# epoch : 0  j = 1  loss = 4.0744205
# epoch : 0  j = 2  loss = 3.991508
# epoch : 0  j = 3  loss = 3.9505157
# epoch : 0  j = 4  loss = 4.0709686
# epoch : 0  j = 5  loss = 4.0701714
# epoch : 0  j = 6  loss = 3.956236
# epoch : 0  j = 7  loss = 4.037812
# epoch : 0  j = 8  loss = 3.9951744
# epoch : 0  j = 9  loss = 3.9643846