#!/usr/bin/env python3

import sympy

def repr_latex_or_repr(o):
    if(hasattr(o, "_repr_latex_")):
        return o._repr_latex_()
    else:
        return repr(o)

class FockState(object):
    def __init__(self, prefactor, n):
        self.n = n
        self.prefactor = prefactor

    def inc(self):
        return FockState(sympy.sqrt(self.n + 1) * self.prefactor, self.n + 1)

    def dec(self):
        if(self.n == 0):
            return ZeroState()
        return FockState(sympy.sqrt(self.n) * self.prefactor, self.n - 1)
    
    def __rmul__(self, other):
        if(isinstance(other, (int, float, sympy.Expr))):
            return FockState(self.prefactor * other, self.n)
        return NotImplemented
    def __matmul__(self, other):
        if(isinstance(other, FockState)):
            if(other.n != self.n):
                return 0
            return sympy.conjugate(self.prefactor) * other.prefactor 
        if(isinstance(other, FockStateAdd)):
            return sum(self @ s for s in other.states)
        return NotImplemented

    def __add__(self, other):
        if(isinstance(other, FockState)):
            return FockStateAdd([self, other])
        return NotImplemented

    def __radd__(self, other):
        if(other == 0):
            return self
        return NotImplemented

    def _repr_latex_(self):
        return repr_latex_or_repr(self.prefactor) + f"$|{self.n}>$"


class ZeroState(FockState):
    def __init__(self):
        self.n = None 
        self.prefactor = 0
    def inc(self):
        return self
    def dec(self):
        return self

    def _repr_latex_(self):
        return " $0$ "

    def __rmul__(self, other):
        return self

class FockStateAdd(object):
    def __init__(self, states):
        self.states = states

    def __add__(self, other):
        if(isinstance(other, FockState)):
            return FockStateAdd(self.states + [other])
        if(isinstance(other, FockStateAdd)):
            return FockStateAdd(self.states + other.states)
        if(other == 0):
            return self
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        if(isinstance(other, FockState)):
            return sum(s @ other for s in self.states)
        if(isinstance(other, FockStateAdd)):
            return sum(s1 @ s2 for s1 in self.states for s2 in other.states)
        return NotImplemented

    def __rmatmul_(self, other):
        if(isinstance(other, FockState)):
           return sum(other @ s for s in self.states)
        return NotImplemented

    def _repr_latex_(self):
        return r"(" + "+".join(s._repr_latex_() for s in self.states) + r")"

    def __rmul__(self, other):
        return FockStateAdd([other * s for s in self.states])
