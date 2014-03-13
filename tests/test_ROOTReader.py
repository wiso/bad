import unittest
import numpy as np
#import atexit

import ROOT

from bad.ROOTReader import GetValuesFromTree


#@atexit.register
#def quite_exit():
#    print "quitting"
#    ROOT.gSystem.Exit(0)

from array import array
tree = ROOT.TTree("tree", "tree")
b1 = array('i', [0])
b2 = array('f', [0.])
tree.Branch('var_i', b1, 'var_i/I')
tree.Branch('var_f', b2, 'var_f/F')
nentries = 1000
for i in xrange(nentries):
    b1[0] = i
    b2[0] = i
    tree.Fill()


class TestReader(unittest.TestCase):

    @classmethod
    def asetUpClass(cls):
        from array import array
        cls.tree = ROOT.TTree("tree", "tree")
        b1 = array('i', [0])
        b2 = array('f', [0.])
        cls.tree.Branch('var_i', b1, 'var_i/I')
        cls.tree.Branch('var_f', b2, 'var_f/F')
        cls.nentries = 1000
        for i in xrange(cls.nentries):
            b1[0] = i
            b2[0] = i
            cls.tree.Fill()

    @classmethod
    def atearDownClass(cls):
#        cls.tree.Delete()
        pass

    def test_numentries(self):
        self.assertEqual(tree.GetEntries(), nentries)
        self.assertEqual(tree.GetListOfBranches().GetEntries(), 2)

    def test_read1D(self):
        x = GetValuesFromTree(tree, "var_i")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int))

        x = GetValuesFromTree(tree, "var_f")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_equal(x, np.arange(1000, dtype=float))

    def test_read1D_formula(self):
        x = GetValuesFromTree(tree, "var_i * 2.")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int) * 2.)

        x = GetValuesFromTree(tree, "sin(var_f * 2.)")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_almost_equal(x, np.sin(np.arange(1000, dtype=float) * 2.))

    def test_read2D_formula(self):
        x, y = GetValuesFromTree(tree, "var_f:var_i * 2.")
        self.assertEqual(len(x), nentries)
        self.assertEqual(len(y), nentries)
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int))
        np.testing.assert_array_equal(y, np.arange(1000, dtype=int) * 2.)
