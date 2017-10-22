#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:47:01 2017

@author: paul
"""
from WeatherClass import Weather
import numpy as np
import SimulatorClass as SimC
from SimulatorClass import Simulator
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib
from matplotlib import animation
matplotlib.rcParams.update({'font.size': 16})



HOURS_TO_DAY = 1 / 24
# %% We load the forecast files
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = './data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)
# %% We shift the times so that all times are in the correct bounds for interpolations
Tini = Wavg.time[0]

Wavg.time = Wavg.time - Tini

# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)
Tf = 24 * 5
times = np.arange(0, Tf * HOURS_TO_DAY, 1 * HOURS_TO_DAY)
lats = np.arange(Wavg.lat[0],Wavg.lat[-1], 0.05)
lons = np.arange(Wavg.lon[0], Wavg.lon[-1], 0.05)

stateInit = [0, 47.5, -3.5 + 360]

Sim = Simulator(times, lats, lons, Wavg, stateInit)

#%%

map2=Sim.preparePlotTraj2(stateInit,Dline=10)
traj=1

for tj in range(traj):
    
    Sim.reset(stateInit)
    
    # we define the action 
    action = 225
    states=[]
    states.append(list(stateInit))
    test=[]
          
    for step in range(len(times)-1) :
        state=Sim.doStep(action)
    #    print('state boucle=' + str(state) +'\n')
        states.append(list(state))
        
#     
#    Sim.plotTraj(states,map)
    Sim.plotTraj(states,map2,quiv=True,scatter=True)
#    Sim.plotTraj(states,map2)
plt.show()
plt.title('Trajectories after ' + str(round(Sim.times[Sim.state[0]]))+ ' days with constant heading of '+ str(action)+ ' deg')

#%%
#writer=animation.FFMpegWriter(bitrate=500)
anim=Sim.animateTraj(Wavg,states,trajSteps=1)
#anim.save('trajAnim2.mp4', fps=5, bitrate=5000, dpi=500)

