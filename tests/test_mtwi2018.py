import unittest
from bender.pretreat import mtwi2018


class TestMTWI2018(unittest.TestCase):
    """test pretreat of MTWI2018."""
    @classmethod
    def test_mtwi2018_hello(cls):
        """hello."""
        result = mtwi2018.hello()
        print(result)
