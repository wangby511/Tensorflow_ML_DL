import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.oxflower17 as oxflower17

X,Y = oxflower17.load_data(one_hot = True)
print("X.shape:",X.shape)
print("Y.shape:",Y.shape)

#Building 'VGG Network'
network = input_data(shape=[None, 227, 227, 3])

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides = 2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides = 2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')

network = regression(network,
                     optimizer = 'rmsprop',
                     loss = 'categorical_crossentropy',
                     learning_rate = 0.001)

#Training
model = tflearn.DNN(network,
                    checkpoint_path = 'model_vgg',
                    max_checkpoints = 1,
                    tensorboard_verbose = 0)
model.fit(X, Y, n_epoch = 10,
          shuffle = True,
          show_metric = True,
          batch_size = 32,
          snapshot_step = 10,
          snapshot_epoch = False,
          run_id = 'vgg_oxflowers17')
model.save('vgg19.tflearn')

# ---------------------------------
# Run id: vgg_oxflowers17
# Log directory: /tmp/tflearn_logs/
# ---------------------------------
# Training samples: 1360
# Validation samples: 0
# --
# Training Step: 1  | time: 43.457s
# | RMSProp | epoch: 001 | loss: 0.00000 - acc: 0.0000 -- iter: 0032/1360
# Training Step: 2  | total loss: 2.55073 | time: 83.226s
# | RMSProp | epoch: 001 | loss: 2.55073 - acc: 0.1406 -- iter: 0064/1360
# 2.79286 | time: 121.603s
# | RMSProp | epoch: 001 | loss: 2.79286 - acc: 0.0511 -- iter: 0096/1360
# Training Step: 4  | total loss: 2.81250 | time: 160.230s
# | RMSProp | epoch: 001 | loss: 2.81250 - acc: 0.0362 -- iter: 0128/1360
# Training Step: 5  | total loss: 2.84849 | time: 198.861s
# | RMSProp | epoch: 001 | loss: 2.84849 - acc: 0.0111 -- iter: 0160/1360
# Training Step: 6  | total loss: 2.85647 | time: 237.641s
# | RMSProp | epoch: 001 | loss: 2.85647 - acc: 0.0040 -- iter: 0192/1360
# Training Step: 7  | total loss: 2.85784 | time: 276.799s
# | RMSProp | epoch: 001 | loss: 2.85784 - acc: 0.0391 -- iter: 0224/1360
# Training Step: 8  | total loss: 2.84047 | time: 316.050s
# | RMSProp | epoch: 001 | loss: 2.84047 - acc: 0.0698 -- iter: 0256/1360
# Training Step: 9  | total loss: 2.83978 | time: 354.701s
# | RMSProp | epoch: 001 | loss: 2.83978 - acc: 0.0990 -- iter: 0288/1360
# Training Step: 10  | total loss: 2.84059 | time: 393.532s
# | RMSProp | epoch: 001 | loss: 2.84059 - acc: 0.0964 -- iter: 0320/1360
# --
# 2.86075 | time: 431.867s
# | RMSProp | epoch: 001 | loss: 2.86075 - acc: 0.0803 -- iter: 0352/1360
# Training Step: 12  | total loss: 2.84418 | time: 471.187s
# | RMSProp | epoch: 001 | loss: 2.84418 - acc: 0.0442 -- iter: 0384/1360
#
# (?, 227, 227, 64)
# (?, 114, 114, 64)
# (?, 57, 57, 128)
# (?, 29, 29, 256)
# (?, 15, 15, 512)
# (?, 8, 8, 512)
# (?, 4096)
# (?, 4096)
# (?, 17)