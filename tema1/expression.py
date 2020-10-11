from itertools import chain


class Variable:
    def __init__(self, *args, **kwargs):
        self.factor = 1
        args = list(args)
        if "Name" in kwargs:
            self.name = kwargs["Name"]
        elif isinstance(args[0], str):
            self.name = args[0]
        else:
            raise AttributeError("Name not found")

        if "Factor" in kwargs:
            self.factor = kwargs["Factor"]
        elif len(args) > 1 and isinstance(args[1], int):
            self.factor = args[1]

    def __mul__(self, other):
        # print("var mul " + self.name)
        if isinstance(other, int):
            self.factor *= other
        else:
            raise TypeError
        return self

    def __rmul__(self, other):
        # print("var rmul " + self.name)
        if isinstance(other, int):
            self.factor *= other
        else:
            raise TypeError
        return self

    def __add__(self, other):
        # print("var add " + self.name)
        return Expression(other, self)

    def __neg__(self):
        return Expression(-1 * self)

    def __sub__(self, other):
        # print("var sub " + self.name)
        return Expression(-other, self)


class Expression:
    def __init__(self, *args, **kwargs):
        # print("Constructing expression: ", args)
        self._internal = Expression.construct(*args)

    def construct(*args):
        class Internal:
            def __init__(self):
                self.variables = None
                self.constant = 0
                self.terms = None

        args = list(args)
        obj = Internal()
        obj.constant = 0
        obj.variables = {}
        obj.terms = [a for a in args if not isinstance(a, Expression)]
        for who in args:
            if isinstance(who, int):
                obj.constant += who
            elif isinstance(who, Variable):
                obj.variables[who.name] = who
            elif isinstance(who, Expression):
                obj.variables = dict(chain.from_iterable(d.items() for d in (obj.variables, who._internal.variables)))
                obj.terms = obj.terms + who._internal.terms
                obj.constant += who._internal.constant
            else:
                raise TypeError
        return obj

    def get_var(self, x):
        if isinstance(x, str):
            name = x
        elif isinstance(x, Variable):
            name = x.name
        else:
            raise TypeError

        if name in self._internal.variables:
            return self._internal.variables[name]
        else:
            return Variable(name, 0)

    def __add__(self, other):
        # print("ex add " + other)
        return Expression(other, self)

    def __sub__(self, other):
        # print("ex sub " + other)
        return Expression(-other, self)

    def __mul__(self, other):
        # print("ex mul " + self.name)
        if isinstance(other, int):
            self._internal.constant *= other
            self._internal.terms = [other * t for t in self._internal.terms]
        else:
            raise TypeError
        return self

    def __rmul__(self, other):
        # print("ex mul " + self.name)
        if isinstance(other, int):
            self._internal.constant *= other
            self._internal.terms = [other * t for t in self._internal.terms]
        else:
            raise TypeError
        return self

    def __neg__(self):
        return -1 * self
