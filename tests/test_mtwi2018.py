# -*- coding: utf-8 -*-

"""Test MTWI2018 module."""

import os
import re
import unittest
import math
import numpy as np

from skimage import io, color, transform
from bender.pretreat import mtwi2018
from bender.io import h5


class TestMTWI2018(unittest.TestCase):
    """Test pretreat of MTWI2018."""

    def test_read_image(self):
        """Test load a image."""
        img = mtwi2018.read_img('./tests/data/mtwi2018/T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.jpg')
        self.assertEqual((737, 737, 3), img.shape)

    def test_read_txt(self):
        """Test load a data of txt file."""
        data = mtwi2018.read_txt('./tests/data/mtwi2018/T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.txt')

        self.assertEqual(21, len(data))
        self.assertLess(0, len(data[0][1]))
        self.assertEqual(8, len(data[0][0]))

    def test_rotation_matrix(self):
        """Test rotate a point."""
        matrix = mtwi2018.rotation_matrix(math.sin(-math.pi/2), math.cos(-math.pi/2), 0.5, 0.5)
        point = matrix.dot(np.array([[1], [1], [1]]))
        self.assertTrue(point.all())

    def test_sort_rectangle_vertices(self):
        """Test sort vertives of a rectangle."""
        result = mtwi2018.sort_rectangle_vertices(np.array([[0, -1], [1, 0], [-1, 0], [0, 1]]))
        self.assertEqual((4, 2), result.shape)
        self.assertTrue((np.array([[-1, 0], [0, 1], [1, 0], [0, -1]]) == result).all())

    def test_calculate_angles(self):
        """Test calculate a angle of two points."""
        result = mtwi2018.calculate_angles(np.array([1, 1]), np.array([0, 0]))
        self.assertEqual(math.atan(1), result)

    def test_compute_bounds(self):
        """Test calculate bounds of two vertices."""
        bounds = mtwi2018.compute_bounds([1, 2.6], [2, 4])
        self.assertTrue(bounds[0] == 1 and bounds[1] == 3 and bounds[2] == 2 and bounds[3] == 4)

    def test_crop_text_img(self):
        """Test crop image with vertices."""
        img = io.imread('./tests/data/mtwi2018/T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.jpg')
        new_img = mtwi2018.crop_text_img(img, [0, 0, 5, 0, 0, 5, 5, 5])
        self.assertEqual((5, 5, 3), new_img.shape)

    @unittest.skip("showing class skipping")
    def test_crop_all_images(self):
        """Test complex function that rotate a rectangle."""
        img_path = './data/train/image_train'
        data_path = './data/train/txt_train'
        count = 1
        words = mtwi2018.load_txt('./data/mtwi2018_words.txt')
        width = 224
        height = 224

        path = './data/mtwi2018.h5'
        h5.create(path)
        h5.create_group(path, 'train')
        h5.create_dataset(path, 'X', '/train', shape=(0, height, width, 3), dtype='int8', maxshape= (None, height, width, 3))
        h5.create_dataset(path, 'Y', '/train', shape=(0, len(words)), dtype='int8', maxshape=(None, len(words)))

        files = os.listdir(img_path)
        labels = np.zeros((100, len(words)))
        imgs = np.zeros((100, width, height, 3))
        pointer = 0
        for file in files:
            print('%d / %d' % (count, len(files)))
            count += 1
            if not os.path.isdir(file):
                img = mtwi2018.read_img(img_path + '/' + file)
                file = re.sub(r'.jpg$', '.txt', file)
                datas = mtwi2018.read_txt(data_path + '/' + file)

                # 过滤gif
                if len(img.shape) > 3:
                    img = img[0]
                # 过滤带有Alpha通道的图
                if img.shape[2] > 3:
                    img = color.rgba2rgb(img)
                for _, item in enumerate(datas):
                    if item[1] == '###':
                        continue

                    text_img = mtwi2018.crop_text_img(np.copy(img), item[0])
                    if text_img.shape[0] == 0 or text_img.shape[1] == 0:
                        continue

                    text_img = transform.resize(text_img, (height, width, 3))
                    # mtwi2018.save_img('./tests/data/%s_%s.jpg' % (file, _), text_img)
                    # 存入数组
                    labels[pointer] = mtwi2018.one_hot(item[1], words)
                    imgs[pointer] = text_img

                    pointer += 1
                    if pointer >= 100:
                        pointer = 0
                        # 写入
                        h5.append(path, labels, '/train/Y')
                        h5.append(path, imgs, '/train/X')
                        # 重置
                        labels = np.zeros((100, len(words)))
                        imgs = np.zeros((100, width, height, 3))

                    self.assertTrue(text_img.shape[0] > 0)

        # 写入
        if pointer > 0:
            h5.append(path, labels[0:pointer], '/train/Y')
            h5.append(path, imgs[0:pointer], '/train/X')

    @unittest.skip("showing class skipping")
    def test_count_words(self):
        """Test count words."""
        data_path = './data/train/txt_train'
        count = 1
        words = {}

        files = os.listdir(data_path)
        for file in files:
            print('%d / %d' % (count, len(files)))
            count += 1
            if not os.path.isdir(file):
                datas = mtwi2018.read_txt(data_path + '/' + file)
                for _, item in enumerate(datas):
                    if item[1] == '###':
                        continue
                    for word in item[1]:
                        words[word] = True

        keys = words.keys()
        l = mtwi2018.load_txt('./data/mtwi2018_words.txt')
        print(len(l))
        path = './data/mtwi2018.h5'
        h5.create(path)
        h5.create_group(path, 'train')
        self.assertTrue(data_path)

    def test_read_words(self):
        path = './data/mtwi2018.h5'
        result = h5.read(path, '/train/Y', 1, 2)
        print(result[:, 0: 10])
