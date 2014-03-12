import unittest
from bad.algebra import *
from bad.algebra import _0, _1, _2, _3, _4, _N


class TestAlgebra(unittest.TestCase):
    def test_simplify_constants(self):
        self.assertEqual((_1 + _2).simplify_constants(), _3)
        self.assertEqual((_1 + _2 + _3 + _0).simplify_constants(), _N(6))
        self.assertEqual(_4.simplify_constants(), _N(4))
