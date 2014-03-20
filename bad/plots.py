import numpy as np
import ROOT


class Binning(object):
    def __init__(self, bins, ranges=None):
        if np.isscalar(bins):
            if bins <= 0:
                raise ValueError("number of bins must be > 0")
            self.nbins = bins
            self.min_value, self.max_value = ranges
            if self.min_value >= self.max_value:
                raise ValueError("min value must be < max value")
            self.binning = np.linspace(self.min_value, self.max_value,
                                       bins + 1, True)
        else:
            if len(bins) <= 0:
                raise ValueError("number of bins must be > 0")
            if np.sorted(bins) != bins:
                raise ValueError("bin edge must be sorted")
            self.nbins = len(bins)
            self.min_value = np.min(bins)
            self.max_value = np.max(bins)
            self.binning = bins


class Plot(object):
    def plot(self, data):
        pass


class Scatter(Plot):
    def __init__(self, linecolor=ROOT.kBlack, linestyle=1, markerstyle=20, markercolor=ROOT.kBlack):
        self.linecolor = linecolor
        self.linestyle = linestyle
        self.markerstyle = markerstyle
        self.markercolor = markercolor

    def plot(self, data):
        print data.shape
        if data.shape[0] == 2:
            raise ValueError("wrong shape of data for scatter plot")
        gr = ROOT.TGraph()
        for i, (x, y) in enumerate(data):
            gr.SetPoint(i, x, y)
        gr.SetLineColor(self.linecolor)
        gr.SetLineStyle(self.linestyle)
        gr.SetMarkerStyle(self.markerstyle)
        gr.SetMarkerColor(self.markercolor)
        return gr


class Histo(Plot):
    def __init__(self, name, title, bins, ranges=None, linecolor=ROOT.kBlack):
        self.name = name
        self.title = title
        self.binning = Binning(bins, ranges)
        self.linecolor = linecolor

    def plot(self, data):
        if data.ndim != 1:
            raise ValueError("wrong shape of data for histogram")
        h = ROOT.TH1F("h", "h", len(self.binning.binning) - 1, self.binning.binning)
        for d in data:
            h.Fill(d)
        h.SetLineColor(self.linecolor)
        return h
