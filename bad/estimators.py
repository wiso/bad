import numpy as np
import numexpr as ne
from functools import wraps
ne.set_num_threads(2)

def errorize(f):
    """
    use this decorator when defining function on data (the mean, the rms, the guassian fit), ...
    to be sure that the output has 2 arguments: the value and the error as a tuple
    if the output is not a tuple of two object, with this decorator the function will
    return (value, 0)
    """
    @wraps(f)  # wraps decorator is needed to preserve function name
    def function(*args, **kwargs):
        result = f(*args, **kwargs)
        if isinstance(result, (list, tuple)) and len(result) == 2:
            return result
        else:
            return (result, 0)
    return function


class Estimator(object):
    name = None
    axis_title = '{0}'
    axis_title_latex = '{0}'

    def __init__(self, use_ne=False):
        self.use_ne = use_ne

    def __call__(self, data, w=None):
        if self.use_ne and getattr(self, "_call_ne", None):
            return self._call_ne(data, w)
        else:
            return self._call(data, w)

    def __repr__(self):
        return '<Estimator {0} at {1}>'.format(self.name, hex(id(self)))

    def axis_name(self, quantity, latex=False):
        if latex:
            return self.axis_title_latex.format(quantity)
        else:
            return self.axis_title.format(quantity)


def create_estimator(f, fne=None, ax_title=None, ax_title_latex=None):

    class C(Estimator):
        name = f.__name__
        axis_title = ax_title or name
        axis_title_latex = ax_title_latex or axis_title

        def _call(self, data, w):
            return f(data, w)

    C.__name__ = f.__name__
    return C


class Mean(Estimator):
    name = 'mean'
    axis_title = '<{0}>'
    axis_title_latex = '<{0}>'

    def _call(self, data, w):
        return np.mean(data)

    def _call_ne(self, data, w):
        return ne.evaluate('sum(data)') / len(data)


class Number(Estimator):
    name = ''

    def _call(self, data, w):
        return self.n

    def _call_ne(self, data, w):
        return self.n

    def __init__(self, n):
        super(Number, self).__init__()
        self.n = n


@create_estimator
def maximum(data, w):
    return np.max(data)


if __name__ == '__main__':
    data = np.arange(1000)
    print Mean()(data)
    print maximum()(data)
    print Number(2)(data)
    print Number(3)(data)

    print Mean()(data)
    print maximum()(data)
    print Number(2)(data)
    print Number(3)(data)

    from timeit import Timer

    for estimator in "Mean()", "Number(2)", "maximum()":
        setup="""
from __main__ import Mean, Number, maximum
import numpy as np
data = np.random.random(10000)
"""
        t = Timer("%s._call(data, None)" % estimator, setup=setup)
        t2 = Timer("%s._call_ne(data, None)" % estimator, setup=setup)

        print estimator, t.timeit(10000), t2.timeit(10000)
