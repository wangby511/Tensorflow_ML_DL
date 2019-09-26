import sys
import time
import numpy as np
import tensorflow as tf



class AlexNet():

    def __init__(self,
                 image_size=227,
                 class_number=751,
                 epoch=10,
                 batch_size=128,
                 learning_rate=0.005,
                 random_seed=2019
                 ):

        self.image_size = image_size
        self.class_number = class_number
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.epoch_test = False

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.input_data = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3], name='input_picture')

            self.input_label = tf.placeholder(tf.float32, shape=[None, self.class_number], name='input_label')

            # print(str(self.input_label.get_shape()))

            """
            part I. CONV + RELU + LRN + MAX_POOL
            (N,227,227,3) -> (N,27,27,96)
            """
            layer1_kernel_size = 11

            layer1_num_filter = 96

            filter_shape = [layer1_kernel_size, layer1_kernel_size, 3, layer1_num_filter]

            self.conv1_w = tf.get_variable('conv1_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv1_b = tf.get_variable('conv1_b', [1, 1, 1, layer1_num_filter], initializer=tf.constant_initializer(0.0))

            # self.conv_filter_layer1 = tf.Variable(tf.truncated_normal([layer1_kernel_size, layer1_kernel_size, 3, 96], dtype=tf.float32, stddev=1e-1))

            self.conv2d_layer1 = tf.nn.conv2d(self.input_data, filter = self.conv1_w, strides = [1, 4, 4, 1], padding = 'SAME')

            self.conv2d_layer1 = self.conv2d_layer1 + self.conv1_b

            self.relu_layer1 = tf.nn.relu(self.conv2d_layer1)

            self.lrn1 = tf.nn.lrn(self.relu_layer1, 5, bias=1, alpha=1e-4, beta=0.75, name="LRN1")

            self.output_layer1 = tf.nn.max_pool(self.lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding = 'VALID')

            # print("output_layer1.get_shape = " + str(self.output_layer1.get_shape()))
            # (batch_size, 27, 27, 96)

            """
            part II. CONV + RELU + LRN + MAX_POOL 
            (N,27,27,96) -> (N,13,13,256)
            """

            layer2_kernel_size = 5

            layer2_num_filter = 256

            filter_shape = [layer2_kernel_size, layer2_kernel_size, layer1_num_filter, layer2_num_filter]

            self.conv2_w = tf.get_variable('conv2_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv2_b = tf.get_variable('conv2_b', [1, 1, 1, layer2_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer2 = tf.nn.conv2d(self.output_layer1, filter=self.conv2_w, strides=[1, 1, 1, 1], padding='SAME')

            self.conv2d_layer2 = self.conv2d_layer2 + self.conv2_b

            self.relu_layer2 = tf.nn.relu(self.conv2d_layer2)

            self.lrn2 = tf.nn.lrn(self.relu_layer2, 5, bias=1, alpha=1e-4, beta=0.75, name="LRN2")

            self.output_layer2 = tf.nn.max_pool(self.lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            # print("output_layer2.get_shape = " + str(self.output_layer2.get_shape()))
            # (batch_size, 13, 13, 256)

            """
            part III. CONV + RELU
            (N,13,13,256) -> (N,13,13,384)
            """
            layer3_kernel_size = 3

            layer3_num_filter = 384

            filter_shape = [layer3_kernel_size, layer3_kernel_size, layer2_num_filter, layer3_num_filter]

            self.conv3_w = tf.get_variable('conv3_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv3_b = tf.get_variable('conv3_b', [1, 1, 1, layer3_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer3 = tf.nn.conv2d(self.output_layer2, filter=self.conv3_w, strides=[1, 1, 1, 1],padding='SAME')

            self.conv2d_layer3 = self.conv2d_layer3 + self.conv3_b

            self.relu_layer3 = tf.nn.relu(self.conv2d_layer3)

            self.output_layer3 = self.relu_layer3

            # print("output_layer3.get_shape = " + str(self.output_layer3.get_shape()))
            # (batch_size, 13, 13, 384)

            """
            layer4 CONV + RELU 
            (N,13,13,384) -> (N,13,13,384)
            """

            layer4_kernel_size = 3

            layer4_num_filter = 384

            filter_shape = [layer4_kernel_size, layer4_kernel_size, layer3_num_filter, layer4_num_filter]

            self.conv4_w = tf.get_variable('conv4_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv4_b = tf.get_variable('conv4_b', [1, 1, 1, layer4_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer4 = tf.nn.conv2d(self.output_layer3, filter=self.conv4_w, strides=[1, 1, 1, 1],padding='SAME')

            self.conv2d_layer4 = self.conv2d_layer4 + self.conv4_b

            self.relu_layer4 = tf.nn.relu(self.conv2d_layer4)

            self.output_layer4 = self.relu_layer4

            # print("output_layer4.get_shape = " + str(self.output_layer4.get_shape()))
            # (batch_size, 13, 13, 384)

            """
            layer5 CONV + RELU + MAX_POOL
            (N,13,13,384) -> (N,6,6,256)
            """

            layer5_kernel_size = 3

            layer5_num_filter = 256

            filter_shape = [layer5_kernel_size, layer5_kernel_size, layer4_num_filter, layer5_num_filter]

            self.conv5_w = tf.get_variable('conv5_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv5_b = tf.get_variable('conv5_b', [1, 1, 1, layer5_num_filter],initializer=tf.constant_initializer(0.1))

            self.conv2d_layer5 = tf.nn.conv2d(self.output_layer4, filter=self.conv5_w, strides=[1, 1, 1, 1],padding='SAME')

            self.conv2d_layer5 = self.conv2d_layer5 + self.conv5_b

            self.relu_layer5 = tf.nn.relu(self.conv2d_layer5)

            self.output_layer5 = tf.nn.max_pool(self.relu_layer5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            # print("output_layer5.get_shape = " + str(self.output_layer5.get_shape()))
            # (batch_size, 6, 6, 256)

            """
            layer6 FC + RELU + DROPOUT
            (N,6,6,256) -> (N,4096)
            
            """
            self.intput_layer6 = tf.reshape(self.output_layer5, [-1, 6 * 6 * 256])

            # print("intput_layer6.get_shape = " + str(self.intput_layer6.get_shape()))
            # (batch_size, 6, 6, 256)

            output_layer5_dim = self.intput_layer6.get_shape().as_list()[1]

            self.fc6_params = tf.get_variable('fc6_params', [output_layer5_dim, 4096], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc6_biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)

            self.fc6 = tf.nn.relu(tf.matmul(self.intput_layer6, self.fc6_params) + self.fc6_biases)

            # self.fc6 = tf.nn.dropout(self.fc6, rate=0.5)

            # print("self.fc6 .get_shape = " + str(self.fc6.get_shape()))
            # (batch_size, 4096)

            """
            layer 7 FC layer2
            (N,4096) -> (N,4096)
            """
            self.fc7_params = tf.get_variable('fc7_params', [4096, 4096], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc7_biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)

            self.fc7 = tf.nn.relu(tf.matmul(self.fc6, self.fc7_params) + self.fc7_biases)

            # self.fc7 = tf.nn.dropout(self.fc7, rate=0.5)

            # print("self.fc7.get_shape = " + str(self.fc7.get_shape()))
            # (batch_size, 4096)

            """
            layer 8 FC layer3
            (N,4096) -> (N,1024)
            """
            self.fc8_params = tf.get_variable('fc8_params', [4096, self.class_number], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc8_biases = tf.Variable(tf.constant(0.0, shape=[self.class_number]), dtype=tf.float32, trainable=True)

            self.fc8 = tf.matmul(self.fc7, self.fc8_params) + self.fc8_biases


            """
            LOSS + OPTIMIZER
            """
            self.ysoft = tf.nn.softmax(self.fc8)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_label, logits=self.fc8))

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            ).minimize(self.loss)

            self.correct_prediction = tf.equal(tf.argmax(self.ysoft, 1), tf.argmax(self.input_label, 1))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def fit(self, X_train, y_train):
        for iter in range(self.epoch):
            for j in range(int(X_train.shape[0]//self.batch_size)):
                start_index = j * self.batch_size
                end_index = min((j + 1) * self.batch_size, X_train.shape[0])
                feed_dict = {
                    self.input_data : X_train[start_index:end_index],
                    self.input_label: y_train[start_index:end_index]
                }
                loss = self.sess.run([self.loss], feed_dict=feed_dict)
                print("epoch :", iter, " j =",j," loss =",loss)
                break

            if self.epoch_test:

                feed_dict = {
                    self.input_data: X_train,
                    self.input_label: y_train
                }
                total_loss, accuracy = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                print("Epoch :",iter," accuracy = ",accuracy * 100,"%")

                # print(y_train[start_index])
                # print("\n\n\n")
            #     for x in range(self.batch_size):
            #         print("y[x] = " + str(y[x]))
            #         print("fc[x] = " + str(fc[x]))
            #         print("ysoft[" + str(x) + "]=" + str(ysoft[x]))
            #         idx = np.argwhere(np.array(y_train[x]) == 1)[0][0]
            #         print('fc[' + str(x) + '][' + str(idx) + ']=' + str(fc[x][idx]))
            # print("\n\n")