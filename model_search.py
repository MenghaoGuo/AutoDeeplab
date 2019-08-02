import torch
import torch.nn as nn
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp (nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):

        super(Cell, self).__init__()
        self.C_out = C
        if C_prev_prev != -1 :
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        if rate == 2 :
            self.preprocess1 = FactorizedReduce (C_prev, C, affine= False)
        elif rate == 0 :
            self.preprocess1 = FactorizedIncrease (C_prev, C)
        else :
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                if C_prev_prev != -1 and j != 0:
                    op = MixedOp(C, stride)
                else:
                    stride = 1
                    op = None
                stride = 1
                self._ops.append(op)
        self.ReLUConvBN = ReLUConvBN (self._multiplier * self.C_out, self.C_out, 1, 1, 0)

    def forward(self, s0, s1, weights):
        if s0 is not None :
            s0 = self.preprocess0 (s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states) if h is not None)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return  self.ReLUConvBN (concat_feature)



