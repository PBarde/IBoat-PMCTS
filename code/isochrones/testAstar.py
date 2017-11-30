# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:57:31 2017

@author: Jean-Michel
"""

import AstarClass as AC
import sys
sys.path.append("../model")
from WeatherClass import Weather
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


#  We load the forecast files
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = '../data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)

#  We shift the times so that all times are in the correct bounds for interpolations
Tini = Wavg.time[0]

Wavg.time = Wavg.time - Tini

#  We set up the parameters of the simulation
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

# We set up the parameters of the simulation : destination

heading = 230

tra = []
for t in Sim.times[0:5]:
    tra.append(list(Sim.doStep(heading)))
destination = copy.copy(Sim.state[1:3])
#destination = [47.45, 356.40]
# %% test unitaires pour la class AstarClass
Sim.reset(stateInit)
solver_iso = AC.Pathfinder(Sim,stateInit[1:3],destination)

# %%
list_voisins = solver_iso.currentvoisin()
for noeud in list_voisins:
    print(noeud)
# doit afficher 6 noeuds

solver_iso.openlist = list_voisins
print(solver_iso.petitfopen())
# doit afficher le noeud de plus petit f parmi la liste précédente

#solver_iso.openlist = [noeud]
#for noeud in list_voisins:
#    solver_iso.ajout_openlist_trie_par_f_et_g(noeud)
#for noeud in solver_iso.openlist:
#    print(noeud)
# doit afficher la liste précédente par ordre de f croissant et si égalité par g décroissant
# %%
noeud_faux = AC.Node(4,5,6)
noeud_vrai = AC.Node(8,list_voisins[3].lat,list_voisins[3].lon)
solver_iso.openlist = list_voisins
fait,noeud_id = solver_iso.testpresopen(noeud_faux)
print(fait)
print(noeud_id)
# doit afficher "false" et "None"

fait,noeud_id = solver_iso.testpresopen(noeud_vrai)
print(fait)
print(noeud_id,'\n',noeud_vrai)
# doit afficher "true" et 2 noeuds avec même lat. et lon. mais temps et val différents
# (le premier à la première liste affichée)

# testpresclose() étant identique à testpresopen(), je ne refais pas les tests

solver_iso.reset()



# %% We do the simulation of isochrone solving
Sim.reset(stateInit)
solver_iso.reset(stateInit[1:3],destination)

politique1 = solver_iso.solver()
print(politique1)

Sim.reset(stateInit)
solver_iso.reset(stateInit[1:3],destination)

#politique2 = solver_iso.solverplus()
#print(politique2)

#le résultat doit être le même pour les 2 politiques mais le temps de calcul peut être pas.
#vitesse max du bateau = 3 m/s ???