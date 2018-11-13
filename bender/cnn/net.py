# -*- coding: utf-8 -*-

"""DNN模块。"""

import tensorflow as tf


def vgg16(x_train, y_train, labels_num):

    # Block1
    conv1 = conv2d(x_train, [3, 3, 3, 64])
    conv2 = conv2d(conv1, [3, 3, 64, 64])
    pool1 = max_pool_2x2(conv2)

    # Block2
    conv3 = conv2d(pool1, [3, 3, 64, 128])
    conv4 = conv2d(conv3, [3, 3, 128, 128])
    pool2 = max_pool_2x2(conv4)

    # Block3
    conv5 = conv2d(pool2, [3, 3, 128, 256])
    conv6 = conv2d(conv5, [3, 3, 256, 256])
    conv7 = conv2d(conv6, [3, 3, 256, 256])
    pool3 = max_pool_2x2(conv7)

    # Block4
    conv8 = conv2d(pool3, [3, 3, 256, 512])
    conv9 = conv2d(conv8, [3, 3, 512, 512])
    conv10 = conv2d(conv9, [3, 3, 512, 512])
    pool4 = max_pool_2x2(conv10)

    # Block5
    conv11 = conv2d(pool4, [3, 3, 512, 512])
    conv12 = conv2d(conv11, [3, 3, 512, 512])
    conv13 = conv2d(conv12, [3, 3, 512, 512])
    pool5 = max_pool_2x2(conv13)

    # Block6
    W_fc14 = weight_variable([7 * 7 * 512, 4096])
    b_fc14 = bias_variable([4096])
    h_pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    relu14 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc14) + b_fc14)

    # Block7
    W_fc15 = weight_variable([4096, 4096])
    b_fc15 = bias_variable([4096])
    relu15 = tf.nn.relu(tf.matmul(relu14, W_fc15) + b_fc15)

    # Block8
    W_softmax16 = weight_variable([4096, labels_num])
    b_softmax16 = bias_variable([labels_num])
    softmax = tf.nn.softmax(tf.matmul(relu15, W_softmax16) + b_softmax16)

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
