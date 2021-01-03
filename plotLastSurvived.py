from CoilClass import Coil
from agent import Agent
import pickle

with open("lastSurvived.pickle", "rb") as file:
    survived = pickle.load(file)

for coil in survived:
    print(coil.distribution)

