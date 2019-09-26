import sys
import time
import numpy as np
import tensorflow as tf



class CaffeNet():

    def __init__(self,
                 image_size=256,
                 class_number=751,
                 epoch=10,
                 batch_size=128,
                 learning_rate=0.005,
                 random_seed=2019,
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

            self.input_data = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3])

            self.input_one_hot_label = tf.placeholder(tf.float32, shape=[self.batch_size, self.class_number])

            """
            part I. CONV + RELU + POOL + LRN
            (batch_size, 256, 256, 3) -> (batch_size, 31, 31, 96)
            """
            layer1_kernel_size = 11

            layer1_num_filter = 96

            filter_shape = [layer1_kernel_size, layer1_kernel_size, 3, layer1_num_filter]

            self.conv1_w = tf.get_variable('conv1_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv1_b = tf.get_variable('conv1_b', [1, 1, 1, layer1_num_filter], initializer=tf.constant_initializer(0.0))

            self.conv2d_layer1 = tf.nn.conv2d(self.input_data, filter = self.conv1_w, strides = [1, 4, 4, 1], padding = 'SAME') + self.conv1_b

            self.relu_layer1 = tf.nn.relu(self.conv2d_layer1)

            self.pool_layer1 = tf.nn.max_pool2d(self.relu_layer1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            self.output_layer1 = tf.nn.lrn(self.pool_layer1, 5, bias=1, alpha=1e-4, beta=0.75)

            print("output_layer1.get_shape = " + str(self.output_layer1.get_shape()))
            # (batch_size, 31, 31, 96)

            """
            part II. CONV + RELU + POOL + LRN 
            (batch_size, 31, 31, 96) -> (batch_size, 15, 15, 256)
            """

            layer2_kernel_size = 5

            layer2_num_filter = 256

            filter_shape = [layer2_kernel_size, layer2_kernel_size, layer1_num_filter, layer2_num_filter]

            self.conv2_w = tf.get_variable('conv2_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv2_b = tf.get_variable('conv2_b', [1, 1, 1, layer2_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer2 = tf.nn.conv2d(self.output_layer1, filter=self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

            self.relu_layer2 = tf.nn.relu(self.conv2d_layer2)

            self.pool_layer2 = tf.nn.max_pool2d(self.relu_layer2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            self.output_layer2 = tf.nn.lrn(self.pool_layer2, 5, bias=1, alpha=1e-4, beta=0.75)

            print("output_layer2.get_shape = " + str(self.output_layer2.get_shape()))
            # (batch_size, 15, 15, 256)

            """
            part III. CONV + RELU
            (batch_size, 15, 15, 256) -> (batch_size, 15, 15, 384)
            """
            layer3_kernel_size = 3

            layer3_num_filter = 384

            filter_shape = [layer3_kernel_size, layer3_kernel_size, layer2_num_filter, layer3_num_filter]

            self.conv3_w = tf.get_variable('conv3_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv3_b = tf.get_variable('conv3_b', [1, 1, 1, layer3_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer3 = tf.nn.conv2d(self.output_layer2, filter=self.conv3_w, strides=[1, 1, 1, 1],padding='SAME') + self.conv3_b

            self.relu_layer3 = tf.nn.relu(self.conv2d_layer3)

            self.output_layer3 = self.relu_layer3

            print("output_layer3.get_shape = " + str(self.output_layer3.get_shape()))
            # (batch_size, 15, 15, 384)

            """
            layer4 CONV + RELU 
            (batch_size, 15, 15, 384) -> (batch_size, 15, 15, 384)
            """

            layer4_kernel_size = 3

            layer4_num_filter = 384

            filter_shape = [layer4_kernel_size, layer4_kernel_size, layer3_num_filter, layer4_num_filter]

            self.conv4_w = tf.get_variable('conv4_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv4_b = tf.get_variable('conv4_b', [1, 1, 1, layer4_num_filter], initializer=tf.constant_initializer(0.1))

            self.conv2d_layer4 = tf.nn.conv2d(self.output_layer3, filter=self.conv4_w, strides=[1, 1, 1, 1],padding='SAME') + self.conv4_b

            self.relu_layer4 = tf.nn.relu(self.conv2d_layer4)

            self.output_layer4 = self.relu_layer4

            print("output_layer4.get_shape = " + str(self.output_layer4.get_shape()))
            # (batch_size, 15, 15, 384)

            """
            layer5 CONV + RELU + POOL
            (batch_size, 15, 15, 384) -> (batch_size, 7, 7, 256)
            """

            layer5_kernel_size = 3

            layer5_num_filter = 256

            filter_shape = [layer5_kernel_size, layer5_kernel_size, layer4_num_filter, layer5_num_filter]

            self.conv5_w = tf.get_variable('conv5_w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.conv5_b = tf.get_variable('conv5_b', [1, 1, 1, layer5_num_filter],initializer=tf.constant_initializer(0.1))

            self.conv2d_layer5 = tf.nn.conv2d(self.output_layer4, filter=self.conv5_w, strides=[1, 1, 1, 1],padding='SAME') + self.conv5_b

            self.relu_layer5 = tf.nn.relu(self.conv2d_layer5)

            self.output_layer5 = tf.nn.max_pool2d(self.relu_layer5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            print("output_layer5.get_shape = " + str(self.output_layer5.get_shape()))
            # (batch_size, 7, 7, 256)

            """
            layer6 FC + RELU + DROPOUT
            (batch_size, 7, 7, 256) -> (batch_size, 7 * 7 * 256) -> (batch_size, 4096)
            
            """
            self.intput_layer6 = tf.reshape(self.output_layer5, [self.batch_size, -1])

            # print("intput_layer6.get_shape = " + str(self.intput_layer6.get_shape()))
            # (batch_size, 12544 = 7 * 7 * 256)

            output_layer5_dim = self.intput_layer6.get_shape().as_list()[1]

            self.fc6_params = tf.get_variable('fc6_params', [output_layer5_dim, 4096], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc6_biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)

            self.fc6 = tf.nn.relu(tf.matmul(self.intput_layer6, self.fc6_params) + self.fc6_biases)

            self.fc6 = tf.nn.dropout(self.fc6, rate=0.5)

            print("self.fc6 .get_shape = " + str(self.fc6.get_shape()))
            # (batch_size, 4096)

            """
            layer 7 FC layer2
            (batch_size, 4096) -> (batch_size, 4096)
            """
            self.fc7_params = tf.get_variable('fc7_params', [4096, 4096], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc7_biases = tf.Variable(tf.constant(0.0, shape=[4096]), dtype=tf.float32, trainable=True)

            self.fc7 = tf.nn.relu(tf.matmul(self.fc6, self.fc7_params) + self.fc7_biases)

            self.fc7 = tf.nn.dropout(self.fc7, rate=0.5)

            print("self.fc7.get_shape = " + str(self.fc7.get_shape()))
            # (batch_size, 4096)

            """
            layer 8-1 FC layer3-1
            (batch_size, 4096) -> (batch_size, self.class_number)
            """
            self.fc8_params_1 = tf.get_variable('fc8_params_1', [4096, self.class_number], tf.float32, tf.random_normal_initializer(0.0, 0.02))

            self.fc8_biases_1 = tf.Variable(tf.constant(0.0, shape=[self.class_number]), dtype=tf.float32, trainable=True)

            self.fc8 = tf.matmul(self.fc7, self.fc8_params_1) + self.fc8_biases_1

            print("self.fc8.get_shape = " + str(self.fc8.get_shape()))
            # (batch_size, self.class_number)

            """
            LOSS I . Identification Loss
            """
            self.y_softmax = tf.nn.softmax(self.fc8)

            self.Identification_Loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_one_hot_label, logits=self.fc8)
            )

            """
            LOSS II . Verification Loss
            batch_size 对半分
            """
            self.half_batch_size = int(self.batch_size/2)

            self.half_fc7_first = self.fc7[0:self.half_batch_size]
            self.half_fc7_second = self.fc7[self.half_batch_size:]

            # print("self.half_fc7_first.get_shape = " + str(self.half_fc7_first.get_shape()))
            # print("self.half_fc7_second.get_shape = " + str(self.half_fc7_second.get_shape()))

            self.input_label = tf.argmax(self.input_one_hot_label, 1)
            # print("self.input_label.get_shape = " + str(self.input_label.get_shape()))

            self.half_label_first =  self.input_label[0:self.half_batch_size]
            self.half_label_second = self.input_label[self.half_batch_size:]

            # print("self.half_label_first.get_shape = " + str(self.half_label_first.get_shape()))
            # print("self.half_label_second.get_shape = " + str(self.half_label_second.get_shape()))

            self.half_fc7_substract = tf.subtract(self.half_fc7_first, self.half_fc7_second)
            # print("self.half_fc7_substract.get_shape = " + str(self.half_fc7_substract.get_shape()))

            self.same_or_diff_label = tf.cast(tf.equal(self.half_label_first, self.half_label_second), tf.int32)
            # print("self.same_or_diff_label.get_shape = " + str(self.same_or_diff_label.get_shape()))

            self.same_or_diff_label_one_hot = tf.one_hot(self.same_or_diff_label, 2)

            """
            layer 8-2 FC layer3-2
            (batch_size/2, 4096) -> (batch_size/2, 2)
            """
            self.fc8_params_2 = tf.get_variable('fc8_params_2', [4096, 2], tf.float32,tf.random_normal_initializer(0.0, 0.02))

            self.fc8_biases_2 = tf.Variable(tf.constant(0.0, shape=[2]), dtype=tf.float32, trainable=True)

            self.fc7_diff = tf.matmul(self.half_fc7_substract, self.fc8_params_2) + self.fc8_biases_2

            # print("self.fc7_diff.get_shape = " + str(self.fc7_diff.get_shape()))
            # (batch_size/2, 2)

            self.Verification_Loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.same_or_diff_label_one_hot, logits=self.fc7_diff)
            )

            self.Joint_Loss = self.Identification_Loss + self.Verification_Loss

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8
            ).minimize(self.Joint_Loss)

            self.correct_prediction = tf.equal(tf.argmax(self.y_softmax, 1), tf.argmax(self.input_label, 1))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.Joint_Loss)

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
                    self.input_one_hot_label: y_train[start_index:end_index]
                }
                loss, input_label = self.sess.run([
                    self.Joint_Loss,
                    self.input_label
                ], feed_dict=feed_dict)
                print("epoch :", iter, " j =",j," loss =",loss)
                # print("half_fc7_first = ",half_fc7_first)
                # print("half_fc7_second = ", half_fc7_second)
                # print("input_label = ", input_label)
                # print("same_or_diff_label = ",same_or_diff_label)
                # print("same_or_diff_label_one_hot = ",same_or_diff_label_one_hot)

            # if self.epoch_test:
            #
            #     feed_dict = {
            #         self.input_data: X_train,
            #         self.input_one_hot_label: y_train
            #     }
            #     total_loss, accuracy = self.sess.run([self.Joint_Loss , self.accuracy], feed_dict=feed_dict)
            #     print("Epoch :",iter," accuracy = ",accuracy * 100,"%")

    # X.shape = (1360, 227, 227, 3)
    # Y.shape = (1360, 17)
    # output_layer1.get_shape = (128, 28, 28, 96)
    # output_layer2.get_shape = (128, 13, 13, 256)
    # output_layer3.get_shape = (128, 13, 13, 384)
    # output_layer4.get_shape = (128, 13, 13, 384)
    # output_layer5.get_shape = (128, 6, 6, 256)
    # self.fc6.get_shape = (128, 4096)
    # self.fc7.get_shape = (128, 4096)
    # self.fc8.get_shape = (128, 17)