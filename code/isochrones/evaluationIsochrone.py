#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:15:51 2018

@author: jean-mi
"""

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
import sys
sys.path.append("../solver")

#from MyTree import Tree
from worker import Tree

# %% We load the forecast files

#ATTENTION il faudra prendre le fichier de vent moyen à terme!!!
mydate = '20180108'
modelcycle = '0100z'
pathToSaveObj = '../data/' + mydate + '_gep_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)
Wavg=Wavg.crop(latBound=[40, 50], lonBound=[360 - 15, 360])

#mydate = '20170519'
#modelcycle = '00'
#pathToSaveObj = '../data/' + mydate + '_' + modelcycle + '.obj'
#Wavg = Weather.load(pathToSaveObj)

# %% We shift the times so that all times are in the correct bounds for interpolations
Tini = Wavg.time[0]

Wavg.time = Wavg.time - Tini

# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)
Tf = 24 * 8
HOURS_TO_DAY = 1/24
times = np.arange(0, Tf * HOURS_TO_DAY, 6 * HOURS_TO_DAY)
lats = np.arange(Wavg.lat[0],Wavg.lat[-1], 0.05)
lons = np.arange(Wavg.lon[0], Wavg.lon[-1], 0.05)

stateInit = [0, 47.5, -3.5 + 360]

SimC.Boat.UNCERTAINTY_COEFF = 0
Sim = Simulator(times, lats, lons, Wavg, stateInit)

# %% We set up the parameters of the simulation : destination

heading = 240
tours = 0
tra = []

for t in Sim.times[0:-1]:
    tours +=1
    tra.append(list(Sim.doStep(heading)))
destination = Sim.state[1:3]

#for t in Sim.times[0:8]:
#    tours +=1
#    tra.append(list(Sim.doStep(heading)))
#destination = Sim.state[1:3]

# %% test déterministe pour la class IsochroneClass
Sim.reset(stateInit)
solver_iso = IC.Isochrone(Sim,stateInit[1:3],destination,delta_cap=10,increment_cap=9,nb_secteur=100,resolution=200)

temps,politique,politique_finale,trajectoire = solver_iso.isochrone_methode() 


print('temps isochrones :',temps) #attention manière de le calculer
print('temps ligne droite :',tours*solver_iso.delta_t) #temps obtenu en faisant la ligne droite (sur 8 pas de temps on est moins bon mais sur la simu plus longue on gagne une demie journée)
#print(politique)
#print(politique_finale)
#print(trajectoire)


states = solver_iso.positions_to_states()
basemap = solver_iso.sim.prepareBaseMap(proj='mill',res='i',Dline=10,dl=1.5,dh=1,centerOfMap=None)
solver_iso.sim.plotTraj(states,basemap,quiv=True)
print(destination)

# %% test stochastique pour la class IsochroneClass
SimC.Boat.UNCERTAINTY_COEFF = 0.4

Sim.reset(stateInit)
liste_states = []
liste_states.append(np.array(stateInit))
for i in range(len((politique))):
    Sim.doStep(politique[i])
    liste_states.append(np.array(Sim.state))

print(Sim.state[1:3])
    
basemap = solver_iso.sim.prepareBaseMap()
solver_iso.sim.plotTraj(liste_states,basemap,quiv=True)
x,y = basemap(destination[0],destination[1])
basemap.scatter(x,y,zorder=3,color='green',label='') #pt mal placé

#%% affichage des isochrones utilisées

basemap = solver_iso.sim.prepareBaseMap()
ind = 0
for isochrone in solver_iso.isochrone_stock:
    for state in isochrone:
        x,y = basemap(state[1],state[2])
        #basemap.plot(x,y,markersize=4,zorder=ind,color='blue')
        basemap.scatter(x,y,zorder=ind,color='green',label='')
        ind += 1

#%% point discriminant
pt_discri = [20,45.9,352.55]
Sim.reset(pt_discri)
solver_iso = IC.Isochrone(Sim,pt_discri[1:3],destination,delta_cap=5,increment_cap=18,nb_secteur=200,resolution=100)

temps,politique,politique_finale,trajectoire = solver_iso.isochrone_methode() 


print('temps isochrones :',temps) #attention manière de le calculer
#temps obtenu en faisant la ligne droite (sur 8 pas de temps on est moins bon mais sur la simu plus longue on gagne une demie journée)
#print(politique)
#print(politique_finale)
#print(trajectoire)


states = solver_iso.positions_to_states()
basemap = solver_iso.sim.prepareBaseMap(proj='mill',res='i',Dline=10,dl=1.5,dh=1,centerOfMap=None)
solver_iso.sim.plotTraj(states,basemap,quiv=True)