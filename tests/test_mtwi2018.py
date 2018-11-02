# -*- coding: utf-8 -*-

"""Test MTWI2018 module."""

import unittest
import math
import numpy as np

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
        self.assertTrue((point == np.array([[1], [-1], [1]])).all())
        
