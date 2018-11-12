# -*- coding: utf-8 -*-

"""DNN模块。"""

import tensorflow as tf


def vgg16(x_train, y_train, labels_num):

    # 第一层
    conv1 = conv2d(x_train, [3, 3, 3, 64])

    # 第二层
    conv2 = conv2d(conv1, [3, 3, 64, 64])

    # 第三层
    pool3 = max_pool_2x2(conv2)

    # 第四层
    conv4 = conv2d(pool3, [3, 3, 64, 128])

    # 第五层
    conv5 = conv2d(conv4, [3, 3, 128, 128])

    # 第六层
    pool6 = max_pool_2x2(conv5)

    # 第七层
    conv7 = conv2d(pool6, [3, 3, 128, 256])

    # 第八层
    conv8 = conv2d(conv7, [3, 3, 256, 256])

    # 第九层
    conv9 = conv2d(conv8, [3, 3, 256, 256])

    # 第十层
    pool10 = max_pool_2x2(conv9)

    # 第十一层
    conv11 = conv2d(pool10, [3, 3, 256, 512])

    # 第十二层
    conv12 = conv2d(conv11, [3, 3, 512, 512])

    # 第十三层
    conv13 = conv2d(conv12, [3, 3, 512, 512])

    # 第十四层
    pool14 = max_pool_2x2(conv13)

    # 第十五
    conv15 = conv2d(pool14, [3, 3, 512, 512])
    conv16 = conv2d(conv15, [3, 3, 512, 512])
    conv17 = conv2d(conv16, [3, 3, 512, 512])

    pool18 = max_pool_2x2(conv17)

    W_fc19 = weight_variable([7 * 7 * 512, 4096])
    b_fc19 = bias_variable([4096])
    h_pool18_flat = tf.reshape(pool18, [-1, 7 * 7 * 512])
    relu19 = tf.nn.relu(tf.matmul(h_pool18_flat, W_fc19) + b_fc19)

    W_fc20 = weight_variable([4096, 4096])
    b_fc20 = bias_variable([4096])
    relu20 = tf.nn.relu(tf.matmul(relu19, W_fc20) + b_fc20)

    W_softmax21 = weight_variable([4096, labels_num])
    b_softmax21 = bias_variable([labels_num])
    softmax = tf.nn.softmax(tf.matmul(relu20, W_softmax21) + b_softmax21)

    # loss
    cross_entropy = -tf.reduce_sum(y_train * tf.log(tf.clip_by_value(softmax,1e-8,1.0)))

    # 预测结果
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_train, 1))
    # 正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return [cross_entropy, accuracy]


def conv2d(input, kernel_size, stride=1):
    kernel = weight_variable(kernel_size)
    bias = bias_variable([kernel_size[3]])
    conv = tf.nn.conv2d(input, kernel, strides=[1, stride, stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bias)
    return tf.nn.relu(conv)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)