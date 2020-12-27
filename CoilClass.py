import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.integrate import quadrature
from numpy import cos, sin, pi, abs, sqrt
import multiprocessing as mp
import math as ma
import datetime as dt
import os
import pickle
import redis
import sys


class Coil():
    def __init__(self, length, minRadius, scWidth, scThickness, stairAmount, layerAmount):
        self.length = length
        self.Z0 = self.length/2
        self.minRadius = minRadius
        self.scWidth = scWidth
        self.scThickness = scThickness
        self.columnAmount = stairAmount
        self.rowAmount = layerAmount  # max turns

        self.distribution = nu.zeros((self.rowAmount, self.columnAmount), dtype=nu.int)
        self.distribution[self.rowAmount//2:, :] = 1
        self.distributionInRealCoordinates = self.calculateDistributionInRealCoordinates()
        self.loss = None


    @classmethod
    def initFromBaseCoil(cls, baseCoil):
        coil = Coil(length=baseCoil.length, minRadius=baseCoil.minRadius, scWidth=baseCoil.scWidth, scThickness=baseCoil.scThickness, stairAmount=baseCoil.columnAmount, layerAmount=baseCoil.rowAmount)
        coil.distribution = baseCoil.distribution.copy()
        coil.distributionInRealCoordinates = baseCoil.distributionInRealCoordinates.copy()
        coil.loss = baseCoil.loss
        return coil


    # for pickle
    def __getstate__(self):
        return {
            'length': self.length,
            'Z0': self.Z0,
            'minRadius': self.minRadius,
            'scWidth': self.scWidth,
            'scThickness': self.scThickness,
            'columnAmount': self.columnAmount,
            'rowAmount': self.rowAmount,
            'distribution': self.distribution.tolist(),
            'distributionInRealCoordinates': self.distributionInRealCoordinates,
            'loss': self.loss
        }

    def __setstate__(self, state):
        self.length = state['length']
        self.Z0 = state['Z0']
        self.minRadius = state['minRadius']
        self.scWidth = state['scWidth']
        self.scThickness = state['scThickness']
        self.columnAmount = state['columnAmount']
        self.rowAmount = state['rowAmount']
        self.distribution = nu.array(state['distribution'])
        self.distributionInRealCoordinates = state['distributionInRealCoordinates']
        self.loss = state['loss']


    def calculateDistributionInRealCoordinates(self):
        zs = nu.linspace(-self.Z0, self.Z0, self.columnAmount).reshape(1, -1) * self.distribution
        rs = nu.linspace(self.minRadius, self.minRadius+self.rowAmount*self.scThickness, self.rowAmount).reshape(-1, 1) * self.distribution
        indices = [ (r, z) for r, z in zip(rs[rs!=0].ravel(), zs[zs!=0].ravel()) ]
        assert len(rs) == len(zs)
        return indices


    def makeDescendant(self, row, column, shouldIncrease):
        coil = Coil.initFromBaseCoil(baseCoil=self)
        if shouldIncrease:
            coil.distribution[row, column] = 1
            coil.distribution[row, -1-column] = 1
        else:
            coil.distribution[row, column] = 0
            coil.distribution[row, -1-column] = 0
        # print(coil.distribution[-2:, :])
        # print(' ')
        return coil


    def makeDescendants(self, amount):
        descendants = []
        count = 0
        amount = amount // 2
        candidates = []
        # set candidates
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation((self.columnAmount+1)//2).tolist()
        else:#even
            candidates = nu.random.permutation(self.columnAmount//2).tolist()
        increasedColumns = set()
        # add increased descendants
        while count <= amount and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[0] == 1:# can't be increased
                continue
            else:# can be increased
                row = nu.where(rows==0)[0][-1]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=True))
                increasedColumns.add(chosenColumn)
                count += 1
        # add decreased descendants
        count = 0
        if self.columnAmount % 2 == 1:#odd
            candidates = nu.random.permutation(list(set(nu.arange((self.columnAmount+1)//2).tolist()) - increasedColumns)).tolist()
        else:#even
            candidates = nu.random.permutation(list(set(nu.arange(self.columnAmount//2).tolist()) - increasedColumns)).tolist()
        decreasedColumns = set()
        while count <= amount and len(candidates) > 0:
            chosenColumn = candidates.pop()
            rows = self.distribution[:, chosenColumn]
            if rows[-1] == 0:# can't be decreased
                continue
            else:# can be decreased
                row = nu.where(rows==1)[0][0]
                descendants.append(self.makeDescendant(row=row, column=chosenColumn, shouldIncrease=False))
                decreasedColumns.add(chosenColumn)
                count += 1

        return descendants


    def calculateL(self):
        # get Ms between all turns
        Ms = nu.zeros((len(self.distributionInRealCoordinates), len(self.distributionInRealCoordinates)))
        for i, (r, z) in enumerate(self.distributionInRealCoordinates):
            for j in range(i, len(self.distributionInRealCoordinates)):
                r_, z_ = self.distributionInRealCoordinates[j]
                Ms[i, j] = MutalInductance(r_, r, d=abs(z-z_)+1e-8)
        Ms += nu.triu(Ms, k=1).T
        return Ms.sum()


def plotBzDistribution(points=50):
        data = pd.read_csv('tempBnormDistribution.csv')
        data = data.pivot(index='r', columns='z', values='B')
        pl.contourf(data.index, data.columns, data.values.T)
        pl.colorbar()
        pl.show()


if __name__ == '__main__':
    plotBzDistribution()
