import time
import numpy as np
import tensorflow as tf



class AlexNet():

    def __init__(self,
                 image_size=227,
                 class_number=751,
                 field_size=2,
                 embedding_size=8,
                 deep_layers=[32, 32],
                 deep_init_size = 50,
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.01,
                 random_seed=2019,
                 ):

        self.image_size = image_size
        self.class_number = class_number
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.input_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3], name='input_picture')

            self.input_label = tf.placeholder(tf.float32, shape=[self.batch_size, self.class_number], name='input_label')

            """
            layer1 CONV + RELU + MAX_POOL(N,227,227,3) -> (N,27,27,96)
            """

            self.filter_layer1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer1 = tf.nn.conv2d(self.input_data, filter = self.filter_layer1, strides = [1, 4, 4, 1], padding = 'VALID')

            self.relu_layer1 = tf.nn.relu(self.conv2d_layer1)

            self.output_layer1 = tf.nn.max_pool(self.relu_layer1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding = 'VALID')

            self.output_layer1 = tf.nn.lrn(self.output_layer1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')



            """
            layer2 CONV + RELU + MAX_POOL (N,27,27,96) -> (N,13,13,256)
            """

            self.filter_layer2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer2 = tf.nn.conv2d(self.output_layer1, filter=self.filter_layer2, strides=[1, 1, 1, 1], padding='SAME')

            self.relu_layer2 = tf.nn.relu(self.conv2d_layer2)

            self.output_layer2 = tf.nn.max_pool(self.relu_layer2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            self.output_layer2 = tf.nn.lrn( self.output_layer2, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn2')

            """
            layer3 CONV + RELU (N,13,13,256) -> (N,13,13,384)
            """

            self.filter_layer3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer3 = tf.nn.conv2d(self.output_layer2, filter=self.filter_layer3, strides=[1, 1, 1, 1], padding='SAME')

            self.output_layer3 = tf.nn.relu(self.conv2d_layer3)

            """
            layer4 CONV + RELU (N,13,13,384) -> (N,13,13,384)
            """

            self.filter_layer4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer4 = tf.nn.conv2d(self.output_layer3, filter=self.filter_layer4, strides=[1, 1, 1, 1],padding='SAME')

            self.output_layer4 = tf.nn.relu(self.conv2d_layer4)

            """
            layer5 CONV + RELU + MAX_POOL (N,13,13,384) -> (N,13,13,256)
            """

            self.filter_layer5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer5 = tf.nn.conv2d(self.output_layer4, filter=self.filter_layer5, strides=[1, 1, 1, 1],padding='SAME')

            self.relu_layer5 = tf.nn.relu(self.conv2d_layer5)

            self.output_layer5 = tf.nn.max_pool(self.relu_layer5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            self.output_layer5 = tf.nn.lrn(self.output_layer5, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn5')

            """
            layer 6 FC layer1
            """
            self.intput_layer6 = tf.reshape(self.output_layer5, [self.batch_size, -1])

            output_layer5_dim = self.intput_layer6.get_shape().as_list()[1]

            self.fc6_params = tf.Variable(tf.truncated_normal([output_layer5_dim, 4096], stddev=1e-1))

            self.fc6 = tf.nn.relu(tf.matmul(self.intput_layer6, self.fc6_params))

            """
            layer 7 FC layer2
            """
            self.fc7_params = tf.Variable(tf.truncated_normal([4096, 4096], stddev=1e-1))

            self.fc7 = tf.nn.relu(tf.matmul(self.fc6, self.fc7_params))

            """
            layer 8 FC layer3
            """
            self.fc8_params = tf.Variable(tf.truncated_normal([4096, self.class_number], stddev=1e-1))

            self.fc8_biases = tf.Variable(tf.constant(-3000.0, shape=[self.class_number]), dtype=tf.float32, trainable=True)

            self.fc8 = tf.nn.relu(tf.matmul(self.fc7, self.fc8_params) + self.fc8_biases)

            """
            """
            self.final_fully_connected_layer = self.fc8

            self.ysoft = tf.nn.softmax(self.fc8)

            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_label, logits=self.final_fully_connected_layer)

            self.loss = tf.reduce_mean(self.loss)

            # self.optimizer = tf.train.AdamOptimizer(
            #     learning_rate=self.learning_rate,
            #     beta1=0.9,
            #     beta2=0.999,
            #     epsilon=1e-8
            # ).minimize(self.loss)

            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def fit(self, X_train, y_train):
        self.epoch = 10
        for iter in range(self.epoch):
            for j in range(int(X_train.shape[0]/self.batch_size)):
                start_index = j * self.batch_size
                end_index = min((j + 1) * self.batch_size, X_train.shape[0])
                feed_dict = {
                    self.input_data : X_train[start_index:end_index],
                    self.input_label: y_train[start_index:end_index]
                }
                y = y_train[start_index:end_index]
                fc, loss, ysoft = self.sess.run([self.final_fully_connected_layer, self.loss, self.ysoft], feed_dict=feed_dict)
                print("epoch :", iter, " j =",j," loss =",loss)
                # for x in range(self.batch_size):
                #     print("y[x] = " + str(y[x]))
                #     print("fc[x] = " + str(fc[x]))
                #     print("ysoft[" + str(x) + "]=" + str(ysoft[x]))
                #     idx = np.argwhere(np.array(y_train[x]) == 1)[0][0]
                #     print('fc[' + str(x) + '][' + str(idx) + ']=' + str(fc[x][idx]))