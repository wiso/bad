import ROOT
import numpy as np
import logging


def FixEstimate(tree):
    """Call tree.SetEstimate(N * tree.GetEntries() ) to avoid problems with
    TTree::Draw N is the maximum number of variables X the maximum
    depth (say 10)"""
    tree.SetEstimate(4 * 10 * tree.GetEntries())


def GetValuesFromTree(tree, variable, cut='', flatten=False, Nevents=-1):
    """
    adapted from PyH4l of B. Lenzi
    GetValuesFromTree(tree, variable, cut) ->
    Return a ndarray with <variable> which can be single or multi-dimensional
    """
    assert tree

    def GetNdim(tree):
        "Return the dimension of the last expression used with TTree::Draw"
        for i in xrange(100):
            if not tree.GetVar(i):
                return i

    def GetData(tree, i, N):
        "Return a ndarray from the i-th expression used with TTree::Draw"
        data = tree.GetVal(i)
        data.SetSize(N)
        return np.fromiter(data, dtype=float)

    t = tree  # tree.Clone()
    FixEstimate(tree)
#     Nevents = int(1e4)
    Ndim = variable.count(":") + 1
    if Ndim == 1:
        N = t.Draw(variable, cut, 'goff')
    else:
        N = t.Draw(variable, cut, 'goff para')

    if N < 0:
        current_file = t.GetCurrentFile()
        msg = None
        if current_file:
            msg = ("problem with formula %s or cut %s with file %s" %
                   (variable, cut, current_file.GetName()))
        else:
            msg = "problem with formula %s or cut %s" % (variable, cut)
        raise ValueError(msg)

    Ndim = GetNdim(t)

    if Ndim == 1:
        return GetData(t, 0, N)
    elif flatten:
        return np.ravel([GetData(t, i, N) for i in range(Ndim)])
    else:
        return np.array([GetData(t, i, N) for i in range(Ndim)])
    # TODO: Return np.ndarray for Ndim > 1.
    # TODO: Split call in pieces of size 4 and merge


def GetValuesFromTreeWithProof(tree, variable, cut='', flatten=False, is1D=False):
    """GetValuesFromTreeWithProof(tree, variable, cut = '', flatten=False, is1D = False)
    Use proof to load branches. Works just like GetValuesFromTree but
    2 subsequent calls will not have aligned values in general"""
    if not is1D and ':' not in variable.replace('::', ''):
        # FIXME: this is correct in general (e.g: a > 0 ? 1 : 0)
        variable = '(1):{0}'.format(variable)  # Draw in 1D case would produce a TH1, we need a TGraph
        is1D = True
    if not hasattr(ROOT, 'gProof'):
        proof = ROOT.TProof.Open('')
        if not proof:
            raise ValueError("Cannot open proof session")
    if not isinstance(tree, ROOT.TChain):
        logging.debug('Converting tree to TChain')
        chain = ROOT.TChain(tree.GetName())
        chain.Add(tree.GetDirectory().GetName())
        tree = chain
    tree.SetProof(True)
    N = tree.Draw(variable, cut)
    output = ROOT.gProof.GetOutputList()[2]
    if isinstance(output, ROOT.TGraph):
        # 1D or 2D case
        y = output.GetX()  # Draw(a:b) gives (b,a) plot
        y.SetSize(N)
        if is1D:
            return np.array(y)
        else:
            x = output.GetY()
            x.SetSize(N)
            a = np.array([x, y])
    elif isinstance(output, ROOT.TPolyMarker3D):
        # 3D case
        x = output.GetP()
        x.SetSize(3 * N)
        # output gives z,y,x values alternated
        a = np.array(x).reshape(N, 3).T
        # z,y,x -> x, y, z
        a = np.flipud(a)
    else:
        raise ValueError('Unrecognised output: %s' % output)
    if flatten:
        return np.ravel(a)
    return a


class ROOTReader(object):
    def __init__(self, filename, treename, quantities, variables, selection=""):
        self.filename = filename
        self.quantities = set(quantities).union(set(variables))
        self.selection = selection

        print self.quantities

        self.file = ROOT.TFile(filename)
        self.tree = self.file.Get(treename)

    def __call__(self):
        result = {}
        for q in self.quantities:
            logging.info("dumping quantity %s", q)
            result[q] = GetValuesFromTree(self.tree, q, cut=self.selection, flatten=True)
        return result
