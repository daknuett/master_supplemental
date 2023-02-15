#!/usr/bin/env python3

import sympy

from .operator_expression import ScalarMul, OpAdd, OpMul, OperatorExpression
from .fockstate import FockStateAdd, FockState
from .abc import Operator

class LadderOperator(Operator):
    def __init__(self, is_raise):
        self.is_raise = is_raise 

    def eval_expression(self, state_or_stateexpr):
        if(isinstance(state_or_stateexpr, FockState)):
            if self.is_raise:
                return state_or_stateexpr.inc()
            return state_or_stateexpr.dec()
        if(isinstance(state_or_stateexpr, FockStateAdd)):
            if(self.is_raise):
                return sum(s.inc() for s in state_or_stateexpr.states)
            return sum(s.dec() for s in state_or_stateexpr.states)
        raise TypeError()

    def __add__(self, other):
        if(isinstance(other, (OperatorExpression, LadderOperator))):
            return OpAdd(self, other)
        return NotImplemented

    def __mul__(self, other):
        if(isinstance(other, (sympy.Expr, int))):
            return ScalarMul(self, other)
        if(isinstance(other, (FockState, FockStateAdd))):
            return self.eval_expression(other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if(isinstance(other, (OperatorExpression, LadderOperator))):
            return OpMul(self, other)
        return NotImplemented

    def _repr_latex_(self):
        if(self.is_raise):
            return r" $a^\dagger$ "
        return " $a$ "

