# -*- coding: utf-8 -*-

"""Test MTWI2018 module."""

import unittest
import math
import numpy as np

from skimage import io, transform
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

    def test_rotate_rectangle(self):
        """Test complex function that rotate a rectangle."""
        img = mtwi2018.read_img('./tests/data/mtwi2018/T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.jpg')
        datas = mtwi2018.read_txt('./tests/data/mtwi2018/T1._WBXtXdXXXXXXXX_!!0-item_pic.jpg.txt')

        for _, item in enumerate(datas):
            points = np.array([
                [item[0][0], item[0][1]],
                [item[0][2], item[0][3]],
                [item[0][4], item[0][5]],
                [item[0][6], item[0][7]]
            ])
            #sort
            points = mtwi2018.sort_rectangle_vertices(points)
            #calculate angles
            top = points[0, :]
            bottom = points[3, :]
            if abs(top[0] - bottom[0]) < 0.01:
                pass
            else:
                radians = mtwi2018.calculate_angles(points[0, :], points[3, :])
                new_img = transform.rotate(img, math.degrees(radians))
                io.imshow(new_img)
                io.show()

