# -*- coding: utf-8 -*-

"""Test MTWI2018 module."""

import os
import unittest
import math
import numpy as np

from skimage import io
from bender.pretreat import mtwi2018


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

    def test_crop_all_images(self):
        """Test complex function that rotate a rectangle."""
        IMG_PATH = './data/train/image_train'
        DATA_PATH = './data/train/txt_train'

        files = os.listdir(IMG_PATH)
        for file in files:
            if not os.path.isdir(file):
                img = mtwi2018.read_img(IMG_PATH + '/' + file)
                datas = mtwi2018.read_txt(DATA_PATH + '/' + file.replace('.jpg.jpg', '.jpg.txt'))
                print(file)
                for _, item in enumerate(datas):
                    if item[1] == '###':
                        continue
                    print(item)
                    text_img = mtwi2018.crop_text_img(np.copy(img), item[0])
                    print(text_img.shape)
                    print()
                    mtwi2018.save_img('./tests/data/%s_%s.jpg' % (file, _), text_img)
                    self.assertTrue(text_img.shape[0] > 0)
