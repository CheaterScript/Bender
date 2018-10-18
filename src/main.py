# -*- coding: utf-8 -*-
import tensorflow as tf
from cnn.net import model
from pretreat import data
from skimage import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# x = tf.Variable(tf.zeros((100,10)), dtype=tf.float32, name = 'X');
# x = 1 + x
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# result = sess.run(1 * x)
# print(result)

# def train():
# 	net = model();
# 	init_op = tf.global_variables_initializer()
# 	with tf.Session() as sess:
# 		sess.run(init_op);
# 		sess.run(net, feed_dist = {x:1});

# file = data.read_txt('E:/DL/Test/data/train/txt_train/T1cpBqXzxhXXXXXXXX_!!0-item_pic.jpg.txt')
# img = data.read_img('E:/DL/Test/data/train/image_train/T1cpBqXzxhXXXXXXXX_!!0-item_pic.jpg.jpg')

# print(item)
# print(points)
# print(data.crop_img(points, img).shape)

def read_data():
	path = "E:/DL/Test/data/train/image_train"
	files = os.listdir(path)
	img_arr = []
	txt_arr = []

	for file in files:
		if not os.path.isdir(file):
			name = os.path.splitext(file)
			img = data.read_img(path + '/' + file)
			txt = data.read_txt("E:/DL/Test/data/train/txt_train" + '/' + name[0] + '.txt')
			for row in txt.itertuples():
				arr = row[1].split(',', 8)
				item = np.array(arr[0:8], dtype=np.float32)
				string = arr[8]
				points = np.array(data.sort_points(item)).reshape((4,3))
				print(points)
				new_img = data.crop_img(points, img)
				# data.imgshow_pd(item, img)
				# print(new_img.shape)
				# plt.imshow(new_img)
				# plt.show()
read_data()

# print(imgs.shape, txts.shape)