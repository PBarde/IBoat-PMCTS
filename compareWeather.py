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
from mpl_toolkits.basemap import Basemap


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
weather025 = Weather.load(pathToSaveObj[0])
weather1 = Weather.load(pathToSaveObj[1])

instant = 40
#weather025.plotMultipleQuiver(otherWeather = weather1)
time = weather025.time[instant]
weather1.Interpolators()
error = np.zeros((len(weather025.lat), len(weather025.lon)))
interp_u = np.zeros((len(weather025.lat), len(weather025.lon)))
interp_v = np.zeros((len(weather025.lat), len(weather025.lon)))

for i, lat in enumerate(weather025.lat):
    for j, lon in enumerate(weather025.lon):
        if lat>=min(weather1.lat) and lat<=max(weather1.lat) and lon>=min(weather1.lon) and lon<=max(weather1.lon):
            query_pt = [time, lat, lon]
            interp_u[i][j] = weather1.uInterpolator(query_pt)
            interp_v[i][j] = weather1.vInterpolator(query_pt)
            error[i][j] = np.sqrt((interp_u[i][j] - weather025.u[instant][i][j])**2 + (interp_v[i][j] - weather025.v[instant][i][j])**2)

        else:
            error[i][j] = float('NaN')
            
#error = np.sqrt((interp_u - weather025.u[instant])**2 + (interp_v - weather025.v[instant])**2)

# %% PLOT
font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
proj='mill'
res='i'
Dline=5
density=1
matplotlib.rc('font', **font)
plt.figure()

m = Basemap(projection=proj, lat_ts=10, llcrnrlon=weather025.lon.min(), \
            urcrnrlon=weather025.lon.max(), llcrnrlat=weather025.lat.min(), urcrnrlat=weather025.lat.max(), \
            resolution=res)

x, y = m(*np.meshgrid(weather025.lon, weather025.lat))
m.pcolormesh(x, y, error, shading='flat', cmap=plt.cm.jet)
scale = 600/21 * weather025.u[instant].max()
m.quiver(x, y, weather025.u[instant], weather025.v[instant], scale = scale)
m.quiver(x,y,interp_u, interp_v, color = 'red', scale = scale)

cbar = m.colorbar(location='right')
cbar.ax.set_ylabel('Magnitude error m/s')
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(weather025.lat[0::Dline], labels=[1, 0, 0, 0])
m.drawmeridians(weather025.lon[0::Dline], labels=[0, 0, 0, 1])
plt.show()

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