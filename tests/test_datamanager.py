import unittest
import numpy as np
from numpy import nan
from bad.DataManager import DataManager
from bad.estimators import Mean


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(5)
        self.data_var1 = np.arange(5) * 10
        self.data_var2 = np.arange(5) * 100

    def test_fill(self):
        dm = DataManager(('axis1', 'axis2'),
                         ((0, 10, 20), (0, 100, 200)),
                         np.object)
        dm.fill(self.data, (self.data_var1, self.data_var2))
        result = np.array([[[], [], [], [], []],
                           [[], [0], [], [], []],
                           [[], [], [1], [], []],
                           [[], [], [], [2, 3, 4], []],
                           [[], [], [], [], []]], dtype=np.object)
        for x, y in zip(result.flat, dm.data().flat):
            self.assertTrue(all(np.array(x) == y))

    def test_apply(self):
        dm = DataManager(('axis1', 'axis2'),
                         ((0, 10, 20), (0, 100, 200)),
                         np.object)
        dm.fill(self.data, (self.data_var1, self.data_var2))
        result = [[nan, nan, nan, nan, nan],
                  [nan, 0., nan, nan, nan],
                  [nan, nan, 1., nan, nan],
                  [nan, nan, nan, 3., nan],
                  [nan, nan, nan, nan, nan]]

        np.testing.assert_array_equal(result, dm.apply(Mean()).data())
if __name__ == '__main__':
    unittest.main()
