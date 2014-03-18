import unittest
import numpy as np

from bad.estimators import *

data = np.random.random(1000)
zeros = np.zeros(1000)
empty = np.array([])
data_nan = np.random.random(1000)
data_nan[100:200] = np.nan


class TestEstimators(unittest.TestCase):
    def test_output_size(self):
        for est in all_estimators:
            r = est()(data)
            self.assertEqual(len(r), 2, msg="estimator %s not returning ntuple of 2 elements: %s" % (est, r))
            self.assertTrue(np.isscalar(r[0]), msg="estimator %s not returning scalar as value: %s" % (est, r[0]))
            self.assertTrue(np.isscalar(r[1]), msg="estimator %s not returning scalar as error: %s" % (est, r[1]))

    def test_output_size_specials(self):
        for d in zeros, data_nan:
            for est in all_estimators:
                r = est()(d)
                self.assertEqual(len(r), 2, msg="estimator %s not return ntuple of 2 elements: %s" % (est, r))
                self.assertTrue(np.isscalar(r[0]), msg="estimator %s not returning scalar as value: %s" % (est, r[0]))
                self.assertTrue(np.isscalar(r[1]), msg="estimator %s not returning scalar as error: %s" % (est, r[1]))

    def test_mean(self):
        m = Mean()
        m1, m2 = m(data)
        self.assertAlmostEqual(m1, np.mean(data))

        m1, m2 = m(zeros)
        self.assertEqual(m1, 0)

        m1, m2 = m(empty)
        self.assertTrue(np.isnan(m1))
        self.assertTrue(np.isnan(m2))

        m1, m2 = m(data_nan)
        self.assertTrue(np.isnan(m1))
        self.assertTrue(np.isnan(m1))

<<<<<<< HEAD
=======

>>>>>>> 1ba77128569b2c0bdd9279d24a5024ae48b6f662
    def test_RMS(self):
        r = Rms()
        r1, r2 = r(data)
        self.assertAlmostEqual(r1, np.std(data))

        r1, r2 = r(zeros)
        self.assertAlmostEqual(r1, 0)

        r1, r2 = r(empty)
        self.assertTrue(np.isnan(r1))
        self.assertTrue(np.isnan(r2))

        r1, r2 = r(data_nan)
        self.assertTrue(np.isnan(r1))
        self.assertTrue(np.isnan(r2))

    def test_truncated(self):
        for est in all_estimators:
            if 'Truncated' in est.__name__:
                if hasattr(est, '_original'):
                    self.assertEqual(est(1.)(data), est._original()(data))
