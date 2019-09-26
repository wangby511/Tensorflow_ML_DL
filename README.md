## Tensorflow_ML_DL
#### self-study

#### git 配置用户名和邮箱方法
##### git config --global user.name "wangby511"
##### git config --global user.email "wangby511@gmail.com"
##### git config -l

#### 关于CaffeNet_tf.py 与 re-id-CaffeNet-pipeline.py 行人重识别项目
##### 本科毕设项目Caffe (2017)
##### tf重新实现版本(2019.9)
##### loss1: identification loss : 原本分类softmax损失
##### loss2: verification loss: batch_size一分为二 提出全连接层向量fc7 两两做差 根据是否是同一行人接入大小为2的softmax层（0代表不同行人，1代表同一行人）
Pedestrian Re-identification Based on Deep Learning
Dec. 2016 - Jun. 2017
• Built deep learning tool of CAFFE to solve pedestrian re-identification mission on Linux using the thought of classifying the pictures of objects to their belongings.
• Made configurations and trained the learning networks on different training datasets. Used the parameters from the trained model to extract the vectors generated in the fully-connected layer to represent every pedestrians. Applied clustering and distance-matching algorithms to match a person in the given pedestrian database and predicted whether the two images of pedestrian belong to the same person.
• Developed a loss-joint function algorithm (identification loss and verification loss) and modify the loss layer to better improve the accuracy of pedestrian re-id result. In Market-1501 dataset, it achieved a 6.7% of improvement in the 1st rank rate separately on AlexNet compared to original networks.
• Modified the type of verification loss based on the idea of siamese network and compared the Softmax type and Contrastive type. Created a demo of interface to show the matching results on MATLAB GUI.

