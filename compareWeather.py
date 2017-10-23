#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:28:21 2017

@author: fabien
"""
from WeatherClass import Weather
from SimulatorClass import Simulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
matplotlib.rcParams.update({'font.size': 16})

#%%
mydate='20171014'
website='http://nomads.ncep.noaa.gov:9090/dods/'
#gfs_0p25/gfs20171014/gfs_0p25_00z'
modelcycle='00'
resolution=['0p25', '1p00']
latBound=[43,50]
lonBound=[-10+360, 360]
url = []
pathToSaveObj = []
urlold = 'http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs20171014/gfs_0p25_00z'
for i in range(2):
    url.append(website + 'gfs_' + resolution[i] + '/gfs' + mydate + '/gfs_' + resolution[i] + '_' + modelcycle + 'z')
    pathToSaveObj.append('./data/' + mydate + '_gfs_' + resolution[i] + '.obj')


# launch in real python console
#Weather.download(url[0],pathToSaveObj[0],latBound=latBound,lonBound=lonBound,timeSteps=[0,81])
#Weather.download(url[1],pathToSaveObj[1],latBound=latBound,lonBound=lonBound,timeSteps=[0,81])

# %%
weathers = []
for i in range(2):
    # We load the forecast files
    weathers.append(Weather.load(pathToSaveObj[i]))

weathers[0].plotMultipleQuiver(otherWeather = weathers[1])



# %% Initialize the two simulators
simulators = []
HOURS_TO_DAY = 1 / 24
timeStep = [1, 6] # in hour
for i in range(2):
    # We load the forecast files
    wthr = Weather.load(pathToSaveObj[i])
    
    # We shift the times so that all times are in the correct bounds for interpolations
    wthr.time = wthr.time - wthr.time[0]
    
    # We set up the parameters of the simulation
    Tf = 24 * 5
    times = np.arange(0, Tf * HOURS_TO_DAY, timeStep[i] * HOURS_TO_DAY)
    lats = np.arange(wthr.lat[0], wthr.lat[-1], 0.05)
    lons = np.arange(wthr.lon[0], wthr.lon[-1], 0.05)
    stateInit = [0, 47.5, -3.5 + 360]
    
    # We create the simulator
    simulators.append(Simulator(times, lats, lons, wthr, stateInit) )

color = ['black','red']
for i in range(2):
    basemap = simulators[i].preparePlotTraj2(stateInit,Dline=10)


    # Fixe the initial state
    simulators[i].reset(stateInit)
    
    # we define the action 
    action = 225
    states=[]
    states.append(list(stateInit))
    test=[]
          
    for step in range(len(simulators[i].times)-1) :
        state=simulators[i].doStep(action)
    #    print('state boucle=' + str(state) +'\n')
        states.append(list(state))
    simulators[i].plotTraj(states,basemap,quiv=True,scatter=True,color = color[i])

plt.show()