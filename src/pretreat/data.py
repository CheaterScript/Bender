import pandas as pd
import math
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
def read_txt(file_name):
	return pd.read_table(file_name, sep='\t', header=None, encoding="utf-8")

def read_img(file_name):
	return io.imread(file_name)

def crop_img(df, img):
	p1 = np.array(df[0,:]).reshape((3,1))
	p2 = np.array(df[1,:]).reshape((3,1))
	p3 = np.array(df[2,:]).reshape((3,1))
	p4 = np.array(df[3,:]).reshape((3,1))
	offset = np.array([[0], [0], [0]])
	if p1[0][0] != p2[0][0]:
		print('bug')
		sin, cos = compute_angles(p2, p1)
		width, height, _ = img.shape
		m = rotation_matrix(sin, cos, width/2, -height/2)
		p1 = np.dot(m, p1)
		p2 = np.dot(m, p2)                                                                                                                                                                                                                                                                                                                        
		p3 = np.dot(m, p3)
		p4 = np.dot(m, p4)
		print(math.acos(cos))
		print(math.asin(sin))
		img = transform.rotate(img, -math.degrees(math.acos(cos)), True)
		shape = img.shape
		offset_x = shape[0] - width 
		offset_y = shape[1] - height
		offset[0][0] = offset_x
		offset[1][0] = offset_y

	p1 = np.rint(p1)
	p4 = np.rint(p4)
	p1 = p1 + offset
	p2 = p2 + offset
	p3 = p3 + offset
	p4 = p4 + offset
	imgshow(p1[0][0], p1[1][0], p3[0][0], p3[1][0], 0, 0, 0, 0, img)
	start_x = int(p1[0][0])
	start_y = int(p1[1][0])
	end_x = int(p4[0][0])
	end_y = int(p4[1][0])
	return img[start_y : end_y, start_x: end_x, :]

def compute_angles(p1, p2):
	p = p1 - p2
	r = np.linalg.norm(p)
	cos = p[0][0] / r
	sin = p[1][0] / r
	return [cos, sin]

def rotation_matrix(sin, cos, tx, ty):
	return np.array([
		[cos, -sin, (1 - cos) * tx - ty * sin],
		[sin, cos, (1 - cos) * -ty - tx * sin],
		[0, 0, 1]
	])

def imgshow(x1, y1, x2, y2, x3, y3, x4, y4, img):
	plt.imshow(img)
	plt.plot(x1, y1, 'o')
	plt.plot(x2, y2, 'o')
	plt.plot(x3, y3, 'o')
	plt.plot(x4, y4, 'o')
	plt.show()

def imgshow_pd(df, img):
	plt.imshow(img)
	plt.plot(df[0], df[1], 'o')
	plt.plot(df[2], df[3], 'o')
	plt.plot(df[4], df[5], 'o')
	plt.plot(df[6], df[7], 'o')
	plt.show()

def sort_points(arr):
	i = 0
	new_arr = []
	while i < 8:
		p = [arr[i], arr[i + 1], 1]
		new_arr.append(p)
		i += 2
	new_arr.sort(key=lambda x:x[1])
	new_arr.sort()
	return new_arr