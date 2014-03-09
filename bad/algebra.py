from functools import total_ordering
from collections import Counter
from operator import mul
import itertools

@total_ordering
class Element(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.order = 1
    def __repr__(self):
        return self.symbol
    def __add__(self, other):
        return GSum(self, other).flat()
    def __mul__(self, other):
        return GMul(self, other).flat()
    def __eq__(self, other):
        return self.symbol == other.symbol
    def __lt__(self, other):  
        return (self.order, self.symbol) < (other.order, other.symbol)
    def flat(self):
        return self
    def simplify(self):
        return self
    def simplify_constants(self):
        return self
    def expand(self):
        return self
    def group(self):
        return self

@total_ordering    
class Number(Element):
    def __init__(self, n):
        super(Number, self).__init__(str(n))
        self.val = n
        self.order = 0
    def __lt__(self, other):
        if type(other) == Number:
            return self.val < other.val
        else:
            return super(Number, self).__lt__(other)
        
_N = Number

class Operation(Element):
    def __init__(self, op_symbol, *args):
        super(Operation, self).__init__("(" + (" %s " % op_symbol).join([x.symbol for x in args]) + ")")
        self.args = args
    def flat(self):
        args = [x.flat() for x in self.args]
        return type(self)(*args)        
    def simplify(self):       
        args = [x.simplify() for x in self.args]
        return type(self)(*args)
    def expand(self):
        args = [x.expand() for x in self.args]
        return type(self)(*args)
        
        
class CommutativeOperation(Operation):
    def __init__(self, op_symbol, *args):
        super(CommutativeOperation, self).__init__(op_symbol, *sorted(args))        
     
       
class GSum(CommutativeOperation):
    def __init__(self, *args):
         super(GSum, self).__init__('+', *args)
         self.order = max([x.order for x in self.args])
    def flat(self):
        args = [x.flat() for x in self.args]
        sums = [x for x in args if type(x) == GSum]
        addenda = []
        for s in sums:
            for a in s.args:
                addenda.append(a)
        others = [x.flat() for x in self.args if type(x) != GSum]
        args = addenda + others
        return GSum(*args)
    def simplify_constants(self):
        args = [x.simplify_constants() for x in self.args]
        numbers = [x for x in args if type(x) == Number]
        other = [x for x in args if type(x) != Number]
        s = Number(sum([n.val for n in numbers]))
        if not other:
            return s
        elif not numbers:
            return GSum(*other)
        else:
            return GSum(s, *other)
    def group(self):
        count = Counter(self.args)
        args = []
        for k, v in count.iteritems():
            if v == 1:
                args.append(k)
            else:
                args.append(GMul(k, Number(v)))
        return GSum(*args)
    def simplify(self):
        super(GSum, self).simplify()
        return self.flat().simplify_constants().group()
        
        
class GMul(CommutativeOperation):
    def __init__(self, *args):
         super(GMul, self).__init__('*', *args)
         self.order = sum([x.order for x in self.args])
    def flat(self):
        prod = [x.flat() for x in self.args if type(x) == GMul]
        factors = []
        for f in prod:
            for a in f.args:
                factors.append(a)
        others = [x.flat() for x in self.args if type(x) != GMul]
        args = factors + others
        return GMul(*args)
    def simplify_constants(self):
        args = [x.simplify_constants() for x in self.args]
        numbers = [x for x in args if type(x) == Number]
        other = [x for x in args if type(x) != Number]
        m = Number(reduce(mul, [n.val for n in numbers], 1))
        if not other:
            return m
        elif not numbers:
            return GMul(*other)
        else:
            return GMul(m, *other)
    def expand(self):
        to_combine = []
        for el in self.args:
            if type(el) == GSum:
                to_combine.append(el.args)
            else:
                to_combine.append([el])
        return GSum(*[GMul(*el) for el in itertools.product(*to_combine)])
    def simplify(self):
        super(GMul, self).simplify()
        return self.flat().simplify_constants().expand()
        
class BinaryOperation(Element):
    def __init__(self, left, right, op_symbol):
        self.left = left
        self.right = right
        super(BinaryOperation, self).__init__("(%s %s %s)" % (left.symbol, op_symbol, right.symbol))
    def simplify(self):
        return self.__class__(self.left.simplify(), self.right.simplify())  
    
          
        

def simplify(el):
    return el.simplify()

        
_0 = _N(0)    
_1 = _N(1)
_2 = _N(2)
_3 = _N(3)
_4 = _N(4)
_x = Element('x')
_y = Element('y')
_w = Element('w')
_z = Element('z')
