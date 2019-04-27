import tflearn
import tflearn.data_utils as du

import tflearn.datasets.mnist as mnist
X,Y,testX,testY = mnist.load_data(one_hot=True)
X      = X.reshape([-1,28,28,1])
testX  = testX.reshape([-1,28,28,1])
X,mean = du.featurewise_zero_center(X)
testY  = du.featurewise_zero_center(testX,mean)

# Building Residual Network
net = tflearn.input_data(shape=[None, 28, 28, 1])
net = tflearn.conv_2d(net, 64, 3, activation ='relu', bias=False)
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net,'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net,10,activation='softmax')

net = tflearn.regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',learning_rate=0.1)
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnet_mnist',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(X, Y, n_epoch=100, validation_set=(testX, testY),
          show_metric=True, batch_size=256, shuffle=True,
          run_id='resnet_mnist')