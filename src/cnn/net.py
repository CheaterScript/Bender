import tensorflow as tf

def model(X = None, Y = None):
	x = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='X')
	y = tf.placeholder(tf.float32, shape=(100), name='Y')

	# 第一层
	conv1 = tf.layers.conv2d(x, 96, kernel_size=11, strides=(4, 4), padding="VALID")
	pool = tf.layers.max_pooling2d(conv1, 5, 1)
	bn1 = tf.layers.batch_normalization(pool)
	relu1 = tf.nn.relu(bn1)

	return relu1