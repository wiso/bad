import logging
import numpy as np
import numexpr as ne
ne.set_num_threads(2)

import ROOT


class Estimator(object):
    name = None
    axis_title = '{0}'
    axis_title_latex = '{0}'

    def __init__(self, use_ne=False):
        self.use_ne = use_ne

    def value(self, data, w=None):
        if self.use_ne and getattr(self, "value_ne", None):
            return self.value_ne(data, w)
        else:
            return self.value_np(data, w)

    def error(self, data, w=None):
        if self.use_ne and getattr(self, "error_ne", None):
            return self.error_ne(data, w)
        elif getattr(self, "error_np", None):
            return self.error_np(data, w)
        else:
            return 0

    def __call__(self, data, w=None):
        result = self.value(data, w)
        len_result = 1
        try:
            len_result = len(result)
        except TypeError:
            pass
        if len_result == 2:
            return result
        else:
            error = 0
            if getattr(self, "error", None):
                error = self.error(data, w)
            return result, error

    def __repr__(self):
        return '<Estimator {0} at {1}>'.format(self.name, hex(id(self)))

    def axis_name(self, quantity, latex=False):
        if latex:
            return self.axis_title_latex.format(quantity)
        else:
            return self.axis_title.format(quantity)


class create_estimator(object):
    def __init__(self, name=None, ax_title=None, ax_title_latex=None):
        self.name = name
        self.axis_title = ax_title
        self.axis_title_latex = ax_title_latex

    def __call__(self, f):  # TODO implement fne
        class C(Estimator):
            name = self.name or f.__name__
            axis_title = self.axis_title or name
            axis_title_latex = self.axis_title_latex or self.axis_title

            @classmethod
            def value_np(cls, data, w=None):
                return f(data, w)

        C.__name__ = f.__name__
        return C


class SmallestInterval(Estimator):
    name = "smallest_interval"
    axis_title = "smallest interval"

    def __init__(self, integral=0.9):
        super(SmallestInterval, self).__init__()
        self.integral = integral

    def value_np(self, data, w=None):
        try:
            N = len(data)
        except TypeError:
            return None
        if not N:
            return None
        if self.integral == 1:
            return 0, len(data)
        if w is not None:
            return min(endpointsForFraction(data, w, self.integral), key=lambda x: x[1] - x[0])
        xsorted = np.sort(data)
        D = np.floor(self.integral * N)
        if D == 0:
            logging.warning("not enought statistics (%d) to compute smallestInterval, data: %s", len(data), str(data))
            return None
        first_index = (xsorted[D:] - xsorted[:-D]).argmin()
        return xsorted[first_index], xsorted[D + first_index]


def get_truncated_data(data, w=None, fraction=0.9):
    if type(data) != np.ndarray:
        data = np.array(data)
    smallest_interval = SmallestInterval(fraction).value_np(data, w)
    if smallest_interval is None:
        return [], None
    indices = np.logical_and(data > smallest_interval[0], data < smallest_interval[1])
    if w is None:
        return data[indices], None
    return data[indices], w[indices]


@create_estimator(ax_title="RMS({0})")
def rms(data, w=None):
    N = len(data)
    if N < 3:
        return np.nan, np.nan
    if w is None:
        std = np.std(data)
    else:
        mean, sumW = np.average(data, weights=w, returned=True)
        mean2 = np.average(data ** 2, weights=w)
        std = np.sqrt(mean2 - mean ** 2)
        N = sumW ** 2 / (w ** 2).sum()  # effective entries
    return std, std / np.sqrt(N - 2)  # TODO: check error formula


class Mean(Estimator):
    name = 'mean'
    axis_title = '<{0}>'
    axis_title_latex = '<{0}>'

    @classmethod
    def value_np(cls, data, w):
        if w is not None:
            value, sumW = np.average(data, weights=w, returned=True)
            N = sumW ** 2 / (w ** 2).sum()  # effective entries
            return value, rms.value_np(data, w) / np.sqrt(N)
        else:
            return np.mean(data), rms.value_np(data)[0] / np.sqrt(len(data))

    def value_ne(self, data, w):
        return ne.evaluate('sum(data)') / len(data)


class Constant(Estimator):
    name = ''

    def value_np(self, data, w):
        return self.n

    def value_ne(self, data, w):
        return self.n

    def __init__(self, n=0):
        super(Constant, self).__init__()
        self.n = n


@create_estimator()
def maximum(data, w=None):
    return np.max(data)


@create_estimator()
def stat(data, w=None):
    result = len(data)
    return result, np.sqrt(result)


@create_estimator()
def sumw(data, w=None):
    if w is None:
        return stat().value_np(data)[0]
    return np.sum(w), np.sqrt(np.square(w).sum())


@create_estimator()
def median(data, w=None):
    if len(data) == 0:
        return np.nan
    if w is None:
        return np.median(data), np.std(data) / np.sqrt(len(data)) * 1.253
    # Find the point that divides the sum of weights in 2
    sort_index = np.argsort(data)
    wSum = np.cumsum(w[sort_index])
    m = 0.5 * wSum[-1]  # sum of weights / 2
    N = 1. * wSum[-1] ** 2 / (w ** 2).sum()  # effective entries
    for i, j in enumerate(wSum):
        if j > m:
            med = 0.5 * (data[sort_index[i]] + data[sort_index[i - 1]])
            return med, rms(data, w)[0] / np.sqrt(N) * 1.253


@create_estimator()
def skew(data, w=None):
    n = len(data)
    if not n:
        return np.nan, 0.
    skew_error = 0
    if n > 8:
        skew_error = np.sqrt(6 * n * (n - 1) / (n - 2) / (n + 1) / (n + 3))
    try:
        from scipy.stats import skew as scipy_skew
    except ImportError:
        mean = np.mean(data)
        try:
            return (np.sum((data - mean) ** 3) / n) / (np.power((np.sum((data - mean) ** 2) / n), 3. / 2.)), skew_error
        except FloatingPointError:
            return np.nan
    else:
        return scipy_skew(data), skew_error


def get_truncated_estimators(original):
    class C(Estimator):
        name = "truncated_" + original.name
        _original = original

        def __init__(self, fraction=0.9):
            super(C, self).__init__()
            self.axis_title = original.axis_title + "_{%f}" % fraction
            self.fraction = fraction

        def value_np(self, data, w=None):
            data, w = get_truncated_data(data, w, self.fraction)
            return original.value_np(data, w)
    C.__name__ = "Truncated" + original.__name__

    return C


TruncatedMean = get_truncated_estimators(Mean)
TruncatedRMS = get_truncated_estimators(rms)
TruncatedMedian = get_truncated_estimators(median)
TruncatedSkew = get_truncated_estimators(skew)


class TruncatedRMSRel(Estimator):
    name = "truncated_rms_rel"

    def __init__(self, fraction=0.9):
        super(TruncatedRMSRel, self).__init__()
        self.fraction = fraction

    def value_np(self, data, w=None):
        rms, rmse = TruncatedRMS(self.fraction)(data, w)
        m, me = TruncatedMean(self.fraction)(data, w)
        value = rms / m
        error = np.sqrt((rmse / rms) ** 2 + (me / m) ** 2) * value
        return value, error


class TailOneSigna(Estimator):
    name = "tail_one_sigma"

    def __init__(self, fraction=0.9):
        super(TailOneSigna, self).__init__()
        self.fraction = fraction

    def value_np(self, data, w=None):
        mean = TruncatedMean(self.fraction)(data, w)
        rms = TruncatedRMS(self.fraction)(data, w)
        cut = mean[0] - rms[0]
        try:
            tail = len(data[data < cut])
            return tail / float(len(data))
        except (ZeroDivisionError, FloatingPointError):
            return np.nan


# TODO: memoize this function
def get_gaussian_fit(data, w=None):
    if len(data) == 0:
        return None
    data = sorted(data)
#    min_value = data[max(0, int(len(data) * 0.01)) ]
#    max_value = data[min(int(len(data) * 0.98) - 1, len(data)-1)]
    min_value = 0.7
    max_value = 1.2
    nbins = max(20, len(data) / 100)
    histo = ROOT.TH1F("histofit", "histofit", nbins, min_value, max_value)
    if w is not None:
        map(histo.Fill, data, w)
    else:
        map(histo.Fill, data)
    center = histo.GetMean()
    sigma = histo.GetRMS()

    fitmin = sigma * (-3) + center
    fitmax = sigma * (+2) + center

    fitsf1 = ROOT.TF1("gaus", "gaus", fitmin, fitmax)

    for i in range(6):
        fitsf1.SetParameter(1, center)
        fitsf1.SetParameter(2, sigma)
        histo.Fit(fitsf1, "0Q", "", fitmin, fitmax)
        center = fitsf1.GetParameter(1)
        sigma = fitsf1.GetParameter(2)
        fitmin = sigma * (-1.) + center
        fitmax = sigma * (+2.5) + center

    if fitsf1 is None:
        return None
    return [fitsf1.GetParameter(i) for i in (0, 1, 2)], [fitsf1.GetParError(i) for i in (0, 1, 2)]


def convoluted_gaussian(data, w=None):
    "convoluted_gaussian(data, w=None)"
    if len(data) == 0:
        return None
    data = sorted(data)
    min_value = data[max(0, int(len(data) * 0.01))]
    max_value = data[min(int(len(data) * 0.98) - 1, len(data) - 1)]
    nbins = max(20, len(data) / 100)
    histo = ROOT.TH1F("histofit", "histofit", nbins, min_value, max_value)
    if w is not None:
        map(histo.Fill, data, w)
    else:
        map(histo.Fill, data)
    center = histo.GetMean()
    sigma = histo.GetRMS()
    fitmin = sigma * (-3) + center
    fitmax = sigma * (+2) + center
    fitsf1 = ROOT.TF1("fitfg",
                      ("[3]*( ( exp( ([1]*[1] + 2*[0]*[2]*x)/(2*[0]*[0]*[2]*[2])) "
                       "* (-1 + sqrt(1/([1]*[1]))*[1]+ "
                       "TMath::Erfc(([1]*[1] + [0] * [2] * ((-1)*[0] + x))/"
                       "(sqrt(2) * [0] * [1] * [2]))))/(2*(-1 + exp(1/[2]) ) * [2] ) )"),
                      fitmin, fitmax)
    fitsf1.SetParameter(0, 1)
    fitsf1.SetParameter(1, 0.005)
    fitsf1.SetParameter(2, 0.003)
    fitsf1.SetParameter(3, 6.5)
    histo.Fit(fitsf1, "0Q", "", fitmin, fitmax)

    if fitsf1 is None:
        return None
    return ([fitsf1.GetParameter(i) for i in (0, 1, 2)],
            [fitsf1.GetParError(i) for i in (0, 1, 2)])


@create_estimator()
def peak_convolution(data, w=None):
    "peak_convolution(data, w=None)"
    try:
        result = convoluted_gaussian(data, w)
        if not result:
            return None
        (peak, sigma, tau), (peak_err, sigma_err, tau_err) = result
        return peak, peak_err
    except FloatingPointError:
        return np.nan


@create_estimator()
def sigma_convolution(data, w=None):
    "sigma_convolution(data, w=None)"
    try:
        result = convoluted_gaussian(data, w)
        if not result:
            return None
        (peak, sigma, tau), (peak_err, sigma_err, tau_err) = result
        return sigma, sigma_err
    except FloatingPointError:
        return np.nan


@create_estimator()
def tau_convolution(data, w=None):
    "tau_convolution(data, w=None)"
    try:
        result = convoluted_gaussian(data, w)
        if not result:
            return None
        (peak, sigma, tau), (peak_err, sigma_err, tau_err) = result
        return tau, tau_err
    except FloatingPointError:
        return np.nan


@create_estimator()
def peak_gaussian(data, w=None):
    "peak_gaussian(data, w=None)"
    gaussian_fit = get_gaussian_fit(data, w)
    if gaussian_fit is None:
        return np.nan
    fit_parameters, fit_parameter_errors = gaussian_fit
    return fit_parameters[1], fit_parameter_errors[1]


@create_estimator()
def width_gaussian(data, w=None):
    "width_gaussian(data, w=None)"
    try:
        fit_parameters, fit_parameter_errors = get_gaussian_fit(data, w)
    except TypeError:
        return np.nan
    return fit_parameters[2], fit_parameter_errors[2]


@create_estimator()
def width_smallest_interval(data, w=None, integral=0.9):
    "width_smallest_interval(data, w=None, integral = 0.9)"
    si = SmallestInterval(integral)(data, None)
    if si is None:
        return np.nan
    width = si[1]
    - si[0]
    # Divide by the number of sigmas (i.e 1 for 0.683, 2 for 0.954, ...)
    Nsigma = ROOT.TMath.ErfInverse(integral) * np.sqrt(2)
    width /= (2 * Nsigma)
    return width, width / np.sqrt(2 * np.floor(integral * len(data)) - 1)


@create_estimator()
def interquartile(data, w=None):
    from scipy.stats.mstats import mquantiles
    q = mquantiles(data, [0.25, 0.75])
    return q[1] - q[0]


# TODO: what is this?
def endpointsForFraction(x, w=None, fraction=0.683, isSorted=False):
    """endpointsForFraction(x, w=None, fraction = 0.683) -->
    Return the endpoints of the intervals that contains at least the given fraction (68.3%)
    of events or weights w if given"""
    # Loop over the sorted values of x with two iterators: for each point x1 in x,
    # the 2nd iterator <it> stops when it reaches the requested fraction
    from itertools import izip
    if w is None:  # set all weights to 1 if not given
        w = [1 for i in x]
    I0 = fraction * sum(w)  # the total fraction
    assert fraction > 0, 'Invalid fraction'
    I = 0  # the fraction between x1 and x2
    if isSorted:
        sorted_values = izip(x, w)
    else:
        sorted_values = sorted(izip(x, w))
    it = iter(sorted_values)
    for x1, w1 in sorted_values:
        while I < I0:
            x2, w2 = it.next()
            I += w2
        yield x1, x2  # I --> yield the fraction for debugging
        I -= w1  # remove the last value for the next iteration


all_estimators = [cls for cls in Estimator.__subclasses__()]


if __name__ == '__main__':
    data = np.arange(1000)

    from timeit import Timer

    for estimator in "Mean()", "Number(2)", "maximum()":
        setup = """
from __main__ import Mean, Number, maximum
import numpy as np
data = np.random.random(10000)
"""
        t = Timer("%s.value_np(data, None)" % estimator, setup=setup)
        t2 = Timer("%s.value_ne(data, None)" % estimator, setup=setup)

        print estimator, t.timeit(10000), t2.timeit(10000)
