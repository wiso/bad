from algebra import Element


class Axis(Element):
    def __init__(self):
        super(Element, self).__init__("Axis")
        self.order = 0


class Canvas(Axis):
    def __init__(self):
        super(Axis, self).__init__("canvas")
        self.order = 0


class Pad(Axis):
    def __init__(self):
        super(Axis, self).__init__("pad")
        self.order = 1


class Color(Axis):
    def __init__(self, value):
        self.color = value
        super(Axis, self).__init__("color <{0}>".format(self.color))
        self.order = 2


class Xaxis(Axis):
    def __init__(self):
        super(Axis, self).__init__("xaxis")
        self.order = 3
