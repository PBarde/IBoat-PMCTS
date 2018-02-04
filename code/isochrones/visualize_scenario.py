import IsochroneClass as IC
import sys

sys.path.append("../model")
from weatherTLKT import Weather
import numpy as np
import simulatorTLKT as SimC
from simulatorTLKT import Simulator
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib
from matplotlib import animation

matplotlib.rcParams.update({'font.size': 16})
import copy
import pickle

sys.path.append("../solver")
import forest

# %% We load the forecast files
website = 'http://nomads.ncep.noaa.gov:9090/dods/'
mydate = '20180130'
modelcycle = '00z'
url = (website + 'gfs_1p00/gfs' + mydate + '/gfs_1p00_' + modelcycle)
pathToSaveObj = '../data/' + mydate + '_gfs_1p00_' + modelcycle + '.obj'
Weather.download(url, pathToSaveObj, latBound=[40, 50], lonBound=[360 - 15, 360])

Wavg = Weather.load(pathToSaveObj)
sim = forest.create_simulators([Wavg], 1)[0]
SimC.Boat.UNCERTAINTY_COEFF = 0
# sim.play_scenario()

stateInit = [0, 42.5, -10 + 360]
destination = [45.5, -8.5 + 360]

# Create the solver
sim.reset(stateInit)
solver_iso = IC.Isochrone(sim, stateInit[1:3], destination, delta_cap=10, increment_cap=9, nb_secteur=100,
                          resolution=200)

temps, politique, politique_finale, trajectoire = solver_iso.isochrone_methode()

print("Temps: {}".format(temps))  # attention manière de le calculer
# print("Temps ligne droite: {}".format(tours * solver_iso.delta_t))  # temps obtenu en faisant la ligne droite (sur 8 pas de temps on est moins bon mais sur la simu plus longue on gagne une demie journée)
print("Politique: {}".format(politique))
print("Trajectoire: {}".format(trajectoire))
print("Destination_: {}".format(destination))
m = sim.prepareBaseMap(centerOfMap=stateInit[1:], proj='aeqd')
for i, el in enumerate(trajectoire):
    trajectoire[i] = [i] + el

sim.plotTraj(trajectoire, m, quiv=True, scatter=True)
