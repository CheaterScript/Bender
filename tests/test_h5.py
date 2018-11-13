# -*- coding: utf-8 -*-

"""Test h5 module."""

import unittest
import numpy as np

from bender.io import h5


class TestH5(unittest.TestCase):
    """Test h5 module."""

    def test_create(self):
        """Test create a HDF5 file."""
        path = './tests/data/test.h5'
        h5.create(path)
        self.assertTrue(path)

    def test_create_group(self):
        """Test create a group."""
        path = './tests/data/test.h5'
        h5.create_group(path, 'data')
        self.assertTrue(path)

    def test_create_dataset(self):
        """Test create a dataset."""
        path = './tests/data/test.h5'
        h5.create_dataset(path, 'X_train', shape=(0, 1, 1, 3),
                          dtype='int32',
                          maxshape=(None, 1, 1, 3)
                         )
        self.assertTrue(path)

    def test_append(self):
        """Test append to a file."""
        path = './tests/data/test.h5'
        h5.append(path, np.zeros((100, 1, 1, 3)), 'X_train')
        h5.append(path, np.zeros((199, 1, 1, 3)), 'X_train')
        self.assertTrue(path)

    def test_read(self):
        """Test read a dataset."""
        path = './tests/data/test.h5'
        h5.read(path, 'X_train', 0, 1)
        self.assertTrue(path)
