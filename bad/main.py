from copy import copy, deepcopy
import ROOT
import logging

logging.basicConfig(level=logging.DEBUG)

# action list

XAXIS = 1
YAXIS = 2
COLOR = 3
SUBPLOT = 4
CANVAS = 5

ACTIONS = (XAXIS, YAXIS, COLOR, SUBPLOT, CANVAS)

"""
  Element("pt", [0,100,300], overflow=True, display=XAXIS) * Element("eta", [0, 1.5, 2.5], display=CANVAS)
  1. -> "eta", "pt"
  2. -> [0, 1.5, 2.5], [0, 100, 300]
  3. ->
"""

# define plotting element base
# -> for normal binning
# -> for equi binning
# -> for file
# -> for algorithm
# -> for cut


class PlottingClass:
    def __init__(self, variable, binning, display):
        self.variable = variable
        self.binning = binning
        self.variables = [variable]
        self.display = display

    def itervariables(self):
        return self.variables

    def __le__(self, rhs):
        return self.display <= rhs.display

    def iterselection(self):
        for i in range(len(self.binning) - 1):
            yield "%f <= %s < %f" % (self.binning[i], self.variable, self.binning[i + 1])

    def iterbinning(self):
        yield {self.variable: self.binning}

    def iterpredicate(self):
        for i in range(len(self.binning) - 1):
            yield self.variable, lambda x: self.binning[i] <= x < self.binning[i + 1]

    def iterbins(self):
        for i in range(len(self.binning) - 1):
            yield (self.binning[i], self.binning[i + 1])

    def itervariablesbins(self):
        for i in range(len(self.binning) - 1):
            yield {self.variable: (self.binning[i], self.binning[i + 1])}

    def __str__(self):
        return self.variable


class PlottingSum:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.variables = set.union(self.p1.variables, self.p2.variables)

    def __str__(self):
        return "(%s + %s)" % (self.p1, self.p2)

    def iterselection(self):
        for s1 in self.p1.iterselection():
            yield s1
        for s2 in self.p2.iterselection():
            yield s2

    def iterbinning(self):
        for s1 in self.p1.iterbinning():
            for s2 in self.p2.iterbinning():
                r = s1
                r.update(s2)
                yield r

    def iterbins(self):
        for s1 in self.p1.iterbins():
            yield s1
        for s2 in self.p2.iterbins():
            yield s2


class PlottingProduct:
    def __init__(self, factors):
        self.factors = sorted(factors)

    def itervariables(self):
        for f in self.factors:
            for v in f.itervariables():
                yield v

    def __str__(self):
        return " x ".join(list(self.itervariables()))

    def iterselection(self):
        for a in [sel.iterselection() for sel in self.factors]:
            print a
            yield "&&".join([str(aa) for aa in a])

    def iterbinning(self):
        for s1 in self.p1.iterbinning():
            for s2 in self.p2.iterbinning():
                r = s1
                r.update(s2)
                yield r

    def iterbins(self):
        for s1 in self.p1.iterbins():
            for s2 in self.p2.iterbins():
                r = copy(s1)
                r.update(s2)
                yield r


class PlottingOrder(object):
    def __init__(self, order):
        self.order = dict()
        if not all([k in ACTIONS for k in order]):
            raise TypeError("not valid action list: %s" % str(order.keys()))
        for action in ACTIONS:
            self.order[action] = order.get(action, None)


def plot(plotting_order):
    logging.debug("creating file output.root")
    output_file = ROOT.TFile("output.root", "RECREATE")
    loop_canvas(plotting_order, output_file)
    output_file.Close()


def loop_canvas(plotting_order, output_file):
    output_file.cd()
    loop = plotting_order.order[CANVAS]
    var = loop.variable
    subloop = deepcopy(plotting_order)
    for bin in loop.iterbins():
        subloop.order[CANVAS].binning = (bin[0], bin[1])
        canvas = ROOT.TCanvas("canvas_%s%s-%s" % (var, bin[0], bin[1]),
                              "canvas %s: [%s, %s]" % (var, bin[0], bin[1]))
        logging.debug("creating canvas %s", canvas.GetName())
        canvas.SetFillColor(2)
        loop_subplot(subloop, canvas)
        canvas.Update()
        canvas.Write()
        canvas.SaveAs(canvas.GetName() + ".png")


def loop_subplot(plotting_order, canvas):
    canvas.cd()
    loop = plotting_order.order[SUBPLOT]
    var = loop.variable
#    subloop = deepcopy(plotting_order)
    bins = list(loop.iterbins())
    # TODO: decide if color / yaxis
    canvas.Divide(1, len(bins))
    canvas.subplots = []
    for i, bin in enumerate(bins):
        logging.debug("creating subplot for %s: [%s, %s]" % (var, bin[0], bin[1]))
        subplot = canvas.cd(i + 1)
        canvas.subplots.append(subplot)
        subplot.SetFillColor(4)
        label = ROOT.TLatex(0.1, 0.9, "%s: [%s, %s]" % (var, bin[0], bin[1]))
        label.SetNDC()
        label.Draw()
        subplot.label = label


def loop_color(plotting_order, subcanvas):
    subcanvas.cd()
    loop = plotting_order.order[COLOR]
#    var = loop.variable
    mg = ROOT.TMultiGraph("multigraph", "multigraph")
    for i, bin in enumerate(loop.iterbins()):
        graph = None
        color = i + 1
        graph.SetLineColor(color)
        mg.Add(graph)
