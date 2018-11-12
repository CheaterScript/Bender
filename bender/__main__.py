"""Main."""

import numpy
import tensorflow as tf

from bender.io import h5
from bender.cnn import net
from bender.pretreat import mtwi2018

def main():
    h5_path = '../data/mtwi2018.h5'
    length = h5.len(h5_path, '/train/X')
    worlds = mtwi2018.load_txt('../data/mtwi2018_words.txt')

    x_train = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='X')
    y_train = tf.placeholder(dtype=tf.float32, shape=[None, len(worlds)], name='Y')

    model, accuracy = net.vgg16(x_train, y_train, len(worlds))
    batch = 32

    # 优化器
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(model)

    with tf.Session() as sess:
        # 初始化公共变量
        sess.run(tf.initialize_all_variables())

        for epoch in range(10):
            current = 0

            for i in range(round(length / batch)):
                start = i * batch
                end = min(start + batch, length)
                batch_x = h5.read(h5_path, '/train/X', start, end).astype('float32') / 255
                batch_y = h5.read(h5_path, '/train/Y', start, end)

                print(numpy.all(batch_y >= 0))

                optimizer.run(feed_dict={x_train: batch_x, y_train: batch_y})
                cost = sess.run(model, feed_dict={x_train: batch_x, y_train: batch_y})
                train_accuracy = accuracy.eval(feed_dict={x_train: batch_x, y_train: batch_y})
                print ("step %d, training accuracy %g, cost: %g" % (i, train_accuracy, cost))


if __name__ == '__main__':
    main()
