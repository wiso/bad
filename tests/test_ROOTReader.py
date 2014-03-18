import unittest
import numpy as np
import atexit

import ROOT

from bad.ROOTReader import GetValuesFromTree, GetValuesFromTreeWithProof, GetValuesFromTreeParallel


@atexit.register
def quite_exit():
    print "quitting"
    ROOT.gSystem.Exit(0)

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

f = ROOT.TFile("tests/test_file.root")
tree_file = f.Get("tree")


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

    def test_read1D_type(self):
        x = GetValuesFromTree(tree, "var_i")
        self.assertEqual(x.dtype, np.int)
        x = GetValuesFromTree(tree, "var_f")
        self.assertEqual(x.dtype, np.float)

    def test_read1D_formula_cut(self):
        x = GetValuesFromTree(tree, "var_i**2", "var_i != 10")
        self.assertEqual(len(x), nentries - 1)
        self.assertEqual(x.shape, (999,))

    def test_read2D_formula(self):
        x, y = GetValuesFromTree(tree, "var_f:var_i * 2.")
        self.assertEqual(type(x), np.ndarray)
        self.assertEqual(type(y), np.ndarray)
        self.assertEqual(len(x), nentries)
        self.assertEqual(len(y), nentries)
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int))
        np.testing.assert_array_equal(y, np.arange(1000, dtype=int) * 2.)

        r = GetValuesFromTree(tree, "var_f:var_i")
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r.shape, (2, nentries))

        r = GetValuesFromTree(tree, "var_f:var_i", flatten=True)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r.shape, (2 * nentries, ))

    def test_readManyD_formula(self):
        r = GetValuesFromTree(tree, "var_f:var_f**2:var_f**3:var_f**4:var_f**5:var_i")
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(len(r), 6)
        self.assertEqual(r.shape, (6, 1000))
        np.testing.assert_array_almost_equal(np.arange(1000), r[0])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 2, r[1])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 3, r[2])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 4, r[3])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 5, r[4])
        np.testing.assert_array_equal(np.arange(1000), r[5])

    def test_invalid_formula(self):
        with self.assertRaises(ValueError):
            GetValuesFromTree(tree, "aa")
        with self.assertRaises(ValueError):
            GetValuesFromTree(tree, "var_i:aa")
        with self.assertRaises(ValueError):
            GetValuesFromTree(tree, "var_i***2")

    def test_numentries_proof(self):
        f = ROOT.TFile("tests/test_file.root")
        t = f.Get("tree")

        r = GetValuesFromTreeWithProof(t, "var_i")
        np.testing.assert_array_almost_equal(np.arange(1000), r)
        self.assertEqual(t.GetEntries(), len(r))

        r = GetValuesFromTreeWithProof(t, "var_i**2")
        np.testing.assert_array_almost_equal(np.arange(1000) ** 2, r)

    def test_read1D_parallel(self):
        nentries = tree_file.GetEntries()
        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_i")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (nentries,))
        np.testing.assert_array_equal(x, np.arange(nentries, dtype=int))

        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (nentries,))
        np.testing.assert_array_equal(x, np.arange(nentries, dtype=float))

    def test_read1D_parallel_formula(self):
        nentries = tree_file.GetEntries()
        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_i * 2.")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int) * 2.)

        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "sin(var_f * 2.)")
        self.assertEqual(len(x), nentries)
        self.assertEqual(x.shape, (1000,))
        np.testing.assert_array_almost_equal(x, np.sin(np.arange(1000, dtype=float) * 2.))

#    def test_wrong_treename_parallel(self):
#        with self.assertRaises(ValueError):
#            GetValuesFromTreeParallel("tests/test_file.root", "WRONG", "var_i * 2.")

    def test_read1D_type_parallel(self):
        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_i")
        self.assertEqual(x.dtype, np.int)
        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f")
        self.assertEqual(x.dtype, np.float)

    def test_read1D_formula_cut_parallel(self):
        x = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_i**2", "var_i != 10")
        self.assertEqual(len(x), nentries - 1)
        self.assertEqual(x.shape, (999,))

    def test_read2D_formula_parallel(self):
        x, y = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f:var_i * 2.")
        self.assertEqual(type(x), np.ndarray)
        self.assertEqual(type(y), np.ndarray)
        self.assertEqual(len(x), nentries)
        self.assertEqual(len(y), nentries)
        np.testing.assert_array_equal(x, np.arange(1000, dtype=int))
        np.testing.assert_array_equal(y, np.arange(1000, dtype=int) * 2.)

        r = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f:var_i")
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r.shape, (2, nentries))

        r = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f:var_i", flatten=True)
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(r.shape, (2 * nentries, ))

    def test_readManyD_formula_parallel(self):
        r = GetValuesFromTreeParallel("tests/test_file.root", "tree", "var_f:var_f**2:var_f**3:var_f**4:var_f**5:var_i")
        self.assertEqual(type(r), np.ndarray)
        self.assertEqual(len(r), 6)
        self.assertEqual(r.shape, (6, 1000))
        np.testing.assert_array_almost_equal(np.arange(1000), r[0])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 2, r[1])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 3, r[2])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 4, r[3])
        np.testing.assert_array_almost_equal(np.arange(1000) ** 5, r[4])
        np.testing.assert_array_equal(np.arange(1000), r[5])
