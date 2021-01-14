import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl


initialCoilAxis = pd.read_csv("./initialCoilBnormDistributionOnAxis.csv", skiprows=8)
initialCoilAxis.columns = ["z", "B"]
bestCoilAxis = pd.read_csv("./bestCoilBnormDistributionOnAxis.csv", skiprows=8)
bestCoilAxis.columns = ["z", "B"]
emptyCoilAxis = pd.read_csv("./emptyCoilBnormDistributionOnAxis.csv", skiprows=8)
emptyCoilAxis.columns = ["z", "B"]

pl.plot(initialCoilAxis.loc[:, "z"], (1 - initialCoilAxis.loc[:, "B"] / emptyCoilAxis.loc[:, "B"])*100, label="Initial Coil")
pl.plot(bestCoilAxis.loc[:, "z"], (1 - bestCoilAxis.loc[:, "B"] / emptyCoilAxis.loc[:, "B"])*100, label="Best Coil")
# pl.yscale("log")
pl.xlabel("Position along Z Axis [m]", fontsize=18)
pl.ylabel("Shielding Rate [%]", fontsize=18)
pl.tick_params(labelsize=14)
pl.legend()
pl.show()


initialCoilTopLine = pd.read_csv("./initialCoilBnormDistributionOnTopLine.csv", skiprows=8)
initialCoilTopLine.columns = ["z", "B"]
bestCoilTopLine = pd.read_csv("./bestCoilBnormDistributionOnTopLine.csv", skiprows=8)
bestCoilTopLine.columns = ["z", "B"]
emptyCoilTopLine = pd.read_csv("./emptyCoilBnormDistributionOnTopLine.csv", skiprows=8)
emptyCoilTopLine.columns = ["z", "B"]

pl.plot(initialCoilTopLine.loc[:, "z"], (1 - initialCoilTopLine.loc[:, "B"] / emptyCoilTopLine.loc[:, "B"])*100, label="Initial Coil")
pl.plot(bestCoilTopLine.loc[:, "z"], (1 - bestCoilTopLine.loc[:, "B"] / emptyCoilTopLine.loc[:, "B"])*100, label="Best Coil")
pl.xlabel("Position along Z Axis [m]", fontsize=18)
pl.ylabel("Shielding Rate [%]", fontsize=18)
pl.tick_params(labelsize=14)
pl.legend()
pl.show()
