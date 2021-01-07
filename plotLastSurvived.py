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
    print(coil.loss)
    print("")

agent = Agent()
coil = survived[0]
coil.loss = None
# print(agent.lossOf(coil))
print(agent.lossOf(agent.initialCoil, writeCoilDistributionPath="initialCoilDistribution.csv", listeningBnormPath="initialCoilBnormDistribution.csv"))
print(agent.lossOf(coil, writeCoilDistributionPath="bestCoilDistribution.csv", listeningBnormPath="bestCoilBnormDistribution.csv"))


exit()

data = pd.read_csv("bestCoilBnormDistribution.csv", skiprows=8)
data.columns = ["r", "z", "B"]
data = data.pivot(index="r", columns="z", values="B")
fig = pl.figure()
pl.contourf(data.index, data.columns, data.values.T)
pl.colorbar()
fig.savefig("bestCoilBnormDistribution.png")
pl.close(fig)


data = pd.read_csv("initialCoilBnormmDistribution.csv", skiprows=8)
data.columns = ["r", "z", "B"]
data = data.pivot(index="r", columns="z", values="B")
fig = pl.figure()
pl.contourf(data.index, data.columns, data.values.T)
pl.colorbar()
fig.savefig("initialCoilBnormDistribution.png")
pl.close(fig)
