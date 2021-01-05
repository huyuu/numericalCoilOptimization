from CoilClass import Coil
import pandas as pd
from agent import Agent
import pickle
from matplotlib import pyplot as pl
from agent import Agent

with open("lastSurvived.pickle", "rb") as file:
    survived = pickle.load(file)

for coil in survived:
    print(coil.distribution)

agent = Agent()
agent.coilDistributionPath = "bestCoilDistribution.csv"

coil = survived[0]
agent.createCoilDistributionFile(survived[0])

data = pd.read_csv("BestCoilBnormDistribution.csv", skiprows=8)
data.columns = ["r", "z", "B"]
data = data.pivot(index="r", columns="z", values="B")
print(data)
fig = pl.figure()
pl.contourf(data.index, data.columns, data.values.T)
pl.colorbar()
fig.savefig("BestCoilBnormDistribution.png")
pl.close(fig)
