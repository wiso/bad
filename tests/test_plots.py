import unittest
import numpy as np
#import atexit

import ROOT

from bad.plots import Scatter, Histo


class TestPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data1D = np.random.normal(0, 1, 100000)
        cls.data2D = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)

    def draw_save(self, obj, filename, options=""):
        canvas = ROOT.TCanvas()
        obj.Draw(options)
        canvas.SaveAs(filename)

    def test_histo(self):
        nbins, m, M = 100, -5, 5
        histo = Histo("histo", "title", nbins, (m, M), ROOT.kBlue)
        h = histo.plot(self.data1D)
        self.draw_save(h, "test_histo.png")

        self.assertEqual(nbins, h.GetNbinsX())
        self.assertEqual(m, h.GetBinLowEdge(1))
        self.assertEqual(M, h.GetBinLowEdge(nbins + 1))
        self.assertEqual(len(self.data1D), h.GetEntries())

    def test_scatter(self):
        scatter = Scatter(markerstyle=2)
        gr = scatter.plot(self.data2D)
        self.draw_save(gr, "test_scatter.png", "AP")
