from itertools import izip

import numpy as np
import numexpr as ne
import ROOT


class DataManager(object):
    def __init__(self, axis_names, binning, dtype=None):
        self.axis_names = axis_names
        self.binning = binning
        shape = [len(b)+2 for b in binning]
        self._data = np.zeros(shape, dtype=dtype)

    def position(self, values):
        if len(values) != self._data.ndim:
            raise ValueError("wrong index dimension: dim(%s) != %d"
                             % (values, self._data.ndim))
        pos = [np.digitize([v], b)[0] for v, b in zip(values, self.binning)]
        return tuple(pos)

    def fill(self, objects, binvalues):
        if len(binvalues) != len(self.binning):
            raise ValueError("number of binning quantity different "
                             "from the number of the axes")
        binnumber = []
        for binvaluesrow, bins in izip(binvalues, self.binning):
            indexes_row = np.digitize(binvaluesrow, bins)
            binnumber.append(indexes_row)
        for pos in np.ndindex(self._data.shape):
            mask = np.ones(len(objects), dtype=bool)
            for p, v in izip(pos, binnumber):
                mask &= (p == v)
            self._data[pos] = objects[mask]

    def __getitem__(self, key):
        return self._data[self.position(key)]

    def __setitem__(self, key, value):
        self._data[self.position(key)] = value

    def data(self):
        return self._data

    def data_noextra(self):
        return self._data[[slice(1, -1)] * self._data.ndim]

    def apply(self, f):
        f = np.vectorize(f)
        new_dm = DataManager(self.axis_names,
                             self.binning,
                             dtype=self._data.dtype)
        new_dm._data = f(self._data)
        return new_dm


class Estimator(object):
    name = None
    axis = "%s"

    def __repr__(self):
        return "<Estimator %s at %s>" % (self.name, hex(id(self)))

    def axis_name(self, quantity):
        return self.axis % quantity


class Mean(Estimator):
    name = "mean"
    axis = "<%s>"

    def __call__(self, data):
        return np.mean(data)

    def call_ne(self, data):
        return ne.evaluate("sum(data)") / len(data)


def histogramming(data, nbins, min, max, name_histo=None, title_histo=None):
    name_histo = name_histo or "histo"
    title_histo = title_histo or name_histo
    h = ROOT.TH1F(name_histo, title_histo, nbins, min, max)
    for i in data:
        h.Fill(i)
    return h
