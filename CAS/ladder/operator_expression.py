#!/usr/bin/env python3

import sympy

from .fockstate import FockStateAdd, FockState, repr_latex_or_repr
from .abc import Operator


class OperatorExpression(object):
    """
    Base Class for operator expressions. All operator expressions must inherite from this class.
    """
    __dispatchers__ = {}
    @classmethod
    def register_dispatcher(cls, operator, expression):
        cls.__dispatchers__.update({operator:expression})

    def __init__(self):
        pass 

    def __add__(self, other):
        if(isinstance(other, OperatorExpression)):
            try:
                return self.__dispatchers__["+"](self, other)
            except:
                return NotImplemented
        return NotImplemented

    def __matmul__(self, other):
        if(isinstance(other, (OperatorExpression, Operator))):
            try:
                return self.__dispatchers__["@"](self, other)
            except:
                return NotImplemented
        return NotImplemented

    def __sub__(self, other):
        if(isinstance(other, OperatorExpression)):
            try:
                return self.__dispatchers__["-"](self, other)
            except:
                return NotImplemented
        return NotImplemented

    def __mul__(self, other):
        if(isinstance(other, (sympy.Expr, int))):
            try:
                return self.__dispatchers__["*"](self, other)
            except:
                return NotImplemented
        if(isinstance(other, (FockState, FockStateAdd))):
            return self.eval_expression(other)
        return NotImplemented

    def __rmul__(self, other):
        if(isinstance(other, (sympy.Expr, int))):
            try:
                return self.__dispatchers__["*"](other, self)
            except:
                return NotImplemented
        return NotImplemented

    def eval_expression(self, state_or_stateexpr):
        pass

    def _repr_latex_(self):
        pass

class ScalarMul(OperatorExpression):
    def __init__(self, o1, o2):
        if(isinstance(o1, (OperatorExpression, Operator))):
            self.scalar = o2
            self.opex = o1
        else:
            self.scalar = o1
            self.opex = o2

    def eval_expression(self, state_or_stateexpr):
        return self.scalar * self.opex.eval_expression(state_or_stateexpr)

    def _repr_latex_(self):
        return repr_latex_or_repr(self.scalar) + self.opex._repr_latex_()

class OpAdd(OperatorExpression):
    def __init__(self, o1, o2):
        self.exprs = [o1, o2]

    def eval_expression(self, state_or_stateexpr):
        return sum(e.eval_expression(state_or_stateexpr) for e in self.exprs)

    def _repr_latex_(self):
        return r"(" + "+".join(e._repr_latex_() for e in self.exprs) + r")"

class OpMul(OperatorExpression):
    def __init__(self, o1, o2):
        self.exprs = [o1, o2]

    def eval_expression(self, state_or_stateexpr):
        for e in reversed(self.exprs):
            state_or_stateexpr = e.eval_expression(state_or_stateexpr)
        return state_or_stateexpr

    def _repr_latex_(self):
        return "".join(e._repr_latex_() for e in self.exprs)

OperatorExpression.register_dispatcher("*", ScalarMul)
OperatorExpression.register_dispatcher("+", OpAdd)
OperatorExpression.register_dispatcher("@", OpMul)
