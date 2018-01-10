# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:55:18 2018

@author: Jean-Michel
"""

import IsochroneClass as IC
import sys
sys.path.append("../model")
from WeatherTLKT import Weather
import numpy as np
import SimulatorTLKT as SimC
from SimulatorTLKT import Simulator
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
import matplotlib
from matplotlib import animation
matplotlib.rcParams.update({'font.size': 16})

import copy
import pickle
import sys
sys.path.append("../solver")

from MyTree import Tree


# %% We load the forecast files
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = '../data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)

# %% We shift the times so that all times are in the correct bounds for interpolations
Tini = Wavg.time[0]

Wavg.time = Wavg.time - Tini

# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)
Tf = 24 * 5
HOURS_TO_DAY = 1/24
times = np.arange(0, Tf * HOURS_TO_DAY, 1 * HOURS_TO_DAY)
lats = np.arange(Wavg.lat[0],Wavg.lat[-1], 0.05)
lons = np.arange(Wavg.lon[0], Wavg.lon[-1], 0.05)

stateInit = [0, 47.5, -3.5 + 360]

SimC.Boat.UNCERTAINTY_COEFF = 0
Sim = Simulator(times, lats, lons, Wavg, stateInit)

# %% We set up the parameters of the simulation : destination

heading = 240

tra = []
for t in Sim.times[0:-1]:
    tra.append(list(Sim.doStep(heading)))
destination = [Sim.state[1:2]]

# %% test unitaires pour la class AstarClass
Sim.reset(stateInit)
solver_iso = IC.Isochrone(Sim,stateInit[1:2],destination)
print(solver_iso.isochrone_actuelle) # liste de 19 noeuds
print(len(solver_iso.isochrone_actuelle)) # 19
print(solver_iso.isochrone_actuelle[0])

solver_iso.isochrone_brouillon()
print(len(solver_iso.isochrone_future)) # 19*19 = 361 noeuds
print(solver_iso.isochrone_future[0])
print(solver_iso.distance_moy_iso)

liste_S,delta_S = solver_iso.secteur_liste()
print(delta_S)
print(liste_S) # liste de 20 secteurs
print(len(liste_S)) # 20
print(liste_S[0])

solver_iso.associer_xij_a_S(liste_S,delta_S)
print(liste_S[0].liste_noeud)
print(liste_S[0].liste_noeud[0])
print(liste_S[0].liste_distance)
print(liste_S[0].liste_distance[0])

solver_iso.nouvelle_isochrone_propre(liste_S)
print(solver_iso.isochrone_actuelle)
print(solver_iso.isochrone_actuelle[0])



# %% We do the simulation of isochrone solving
Sim.reset(stateInit)
solver_iso.reset(stateInit[1:2],destination)

temps,politique,trajectoire = solver_iso.isochrone_methode()
print(temps)
print(politique)
print(trajectoire)

