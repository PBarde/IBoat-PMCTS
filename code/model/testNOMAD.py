#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:13:45 2017
0:-1:density
@author: paul
"""

# basic NOMADS OpenDAP extraction and plotting script
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os
from WeatherClass import Weather
import pickle
import math
import scipy.sparse as sparse
import scipy.io as sio
#from MDPClass import MDP
import netCDF4
# set up the figure
#%%

#
#coeffsVel=[-0.0310198207067136,	0.0106968590293653,	-0.0600881995286002,	-0.000370260554113750,	0.00665508747173206,	0.0286695485969272,
#           4.94175336099537e-06,	-7.68003222256045e-05,	-4.03686836415123e-05,	-0.00684406929296715,	-2.91900968067076e-08,	5.68757177397705e-07,
#           2.44425618051996e-06,	2.38962919033178e-05,	0.000379636836557256,	6.34463723894578e-11,	-1.67287818401469e-09,	1.17646387245609e-08,	-2.16961875441841e-08,
#           -6.16724919464073e-07,	-6.77704610076153e-06]
#
#coeffVel=[[-0.03102,-0.06009,0.02867, -0.006844, 0.0003796, -6.777e-06], \
#          [0.0107, 0.006655, -4.037e-05, 2.39e-05, -6.167e-007,0], \
#          [-0.0003703, -7.68e-05, -2.444e-06, -2.17e-08,0,0],\
#          [4.942e-06, 5.688e-07, 1.176e-08,0,0,0],\
#          [-2.919e-08, -1.673e-09,0,0,0,0],\
#          [6.345e-11,0,0,0,0,0]]

#coeffVel2=[[-0.0310198207067136,-0.0600881995286002,0.0286695485969272, -0.00684406929296715, 0.000379636836557256, -6.77704610076153e-06], \
#          [0.0106968590293653, 0.00665508747173206, -4.03686836415123e-05, 2.38962919033178e-05, -6.16724919464073e-07,0], \
#          [-0.000370260554113750, -7.68003222256045e-05, -2.44425618051996e-06, -2.16961875441841e-08,0,0],\
#          [4.94175336099537e-06, 5.68757177397705e-07, 1.17646387245609e-08,0,0,0],\
#          [-2.91900968067076e-08, -1.67287818401469e-09,0,0,0,0],\
#          [6.34463723894578e-11,0,0,0,0,0]]

FitVelocity=((-0.0310198207067136,-0.0600881995286002,0.0286695485969272, -0.00684406929296715, 0.000379636836557256, -6.77704610076153e-06), \
          (0.0106968590293653, 0.00665508747173206, -4.03686836415123e-05, 2.38962919033178e-05, -6.16724919464073e-07,0), \
          (-0.000370260554113750, -7.68003222256045e-05, -2.44425618051996e-06, -2.16961875441841e-08,0,0),\
          (4.94175336099537e-06, 5.68757177397705e-07, 1.17646387245609e-08,0,0,0),\
          (-2.91900968067076e-08, -1.67287818401469e-09,0,0,0,0),\
          (6.34463723894578e-11,0,0,0,0,0))

np.polynomial.polynomial.polyval2d(180,19,FitVelocity)

surf=np.polynomial.polynomial.polygrid2d(np.arange(180),np.arange(20),FitVelocity)
X,Y=np.meshgrid(np.arange(180),np.arange(20), indexing='ij')

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, surf)
ax.set_xlabel('Point of sail')
ax.set_ylabel('wind mag')
ax.set_zlabel('Boat speed')

FitPhi = ((-1.45853032453996 , 4.93261306678728, 2.11068745908768,-0.146421823552765 , 0.00571577488236430,-3.86548180668075e-05 ),\
          ( 0.0450146899230347,-0.265143346808179 , -0.0158879828373159, -6.98702298810359e-05, -2.23691346349847e-05,0 ),\
          ( -6.92983512351375e-05,0.00438845837953799 , 0.000149659040461729, 4.59832959915573e-06, 0,0 ),\
          ( -8.07833836390101e-06, -3.22527331822989e-05,-7.32725492793235e-07 , 0, 0, 0),\
          ( 9.32065359630392e-08, 8.58978734563361e-08, 0, 0, 0, 0),\
          ( -2.99522066048220e-10, 0, 0, 0, 0, 0))

np.polynomial.polynomial.polyval2d(180,19,FitPhi)

surf2=np.polynomial.polynomial.polygrid2d(np.arange(180),np.arange(20),FitPhi)


fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, surf2)
ax.set_xlabel('Point of sail')
ax.set_ylabel('wind mag')
ax.set_zlabel('Boat heeling angle')
#%% Download Normal GFS files

mydate='20170519'
website='http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs'
modelcycle='00'
resolution='0p25'
url=website+mydate+'/gfs_'+resolution+'_'+modelcycle+'z'
#url='http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww320170519/nww320170519_00z'
#os.mkdir('./data/'+mydate)
latBound=[43,50]
lonBound=[-10+360, 360]
pathToSaveObj='./data/'+mydate+'_'+modelcycle+'2.obj'
Weather.download(url,pathToSaveObj,latBound=latBound,lonBound=lonBound,timeSteps=[0,81])

#%%
mydate='20170427'
website='http://nomads.ncep.noaa.gov:9090/dods/gens_bc/gens'
modelcycle='00'

url=website+mydate+'/gespr_'+modelcycle+'z'
#url='http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww320170519/nww320170519_00z'
#os.mkdir('./data/'+mydate)
latBound=[43,50]
lonBound=[-10+360, 360]
timeSteps=[0,65]
#pathToSaveObj='./data/'+mydate+'_'+modelcycle+'ens.obj'
pathToSaveObj='./data/'+mydate+'_old.obj'
#%%
Weather.download(url,pathToSaveObj,latBound=latBound,lonBound=lonBound,ens=True)
#%%
file = netCDF4.Dataset(url)
lat  = file.variables['lat'][:]
lon  = file.variables['lon'][:]
time = file.variables['time'][timeSteps[0]:timeSteps[1]]

latli = np.argmin( np.abs( lat - latBound[0] ) )
latui = np.argmin( np.abs( lat - latBound[1] ) ) 

# longitude lower and upper index
lonli = np.argmin( np.abs( lon - lonBound[0] ) )
lonui = np.argmin( np.abs( lon - lonBound[1] ) )
lat=lat[latli:latui]
lon=lon[lonli:lonui]

u = file.variables['ugrd10m'][:,timeSteps[0]:timeSteps[1],latli:latui,lonli:lonui]
v = file.variables['vgrd10m'][:,timeSteps[0]:timeSteps[1],latli:latui,lonli:lonui]
#%%
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
mydate='20170519'
url='http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww3'+ \
    mydate+'/nww3'+mydate+'_00z'
file = netCDF4.Dataset(url)
lat  = file.variables['lat'][:]
lon  = file.variables['lon'][:]
data = file.variables['htsgwsfc'][1,:,:]
file.close()

#%%
Weather.download(url,'./test.obj')
#%%
latBound=[20,50]
lonBound=[-80+360, 360]

#latBound=[43,50]
#lonBound=[-10+360, 360]

#latBound=[50-24,50]
#lonBound=[-24+360, 360]
nbTimes=5
#%%
W=Weather.load(pathToSaveObj)
#%%
W.animateQuiver(res='c',Dline=100,density=10)
#%%

W=Weather.load(pathToSaveObj,latBound,lonBound)

#%%
timeSteps=np.arange(7)
W=W.crop(latBound,lonBound)
W2=W.crop(latBound,lonBound,timeSteps)
#%%
latI=np.arange(W.lat[0],W.lat[-1],0.05)
lonI=np.arange(W.lon[0],W.lon[-1],0.05)
W.spaceInterpolate(latI,lonI)
W.plotColorQuiver(Dline=25,density=5)
W2.plotColorQuiver()
#%%
time=np.arange(W.time[0],W.time[-1],0.05)
test=griddata(time,W.time,W.u)
#%%
W.getPolarVel()
W.animateQuiver(proj='mill', res='i', instant=0, Dline=25,density=5)
#%%
DgridLat=abs((W.lat[1]-W.lat[0])/2)
W.latGrid=np.empty(np.shape(W.lat))
W.latGrid[0]=W.lat[0]-DgridLat
W.latGrid[0:]=W.lat+DgridLat
         
DgridLon=abs((W.lon[1]-W.lon[0])/2)
W.lonGrid=np.empty(np.shape(W.lon))
W.lonGrid[0]=W.lon[0]-DgridLon
W.lonGrid[0:]=W.lon+DgridLon

#%%
latBound=[45,50]
lonBound=[-10+360, 360-2]
W1=W.crop(latBound,lonBound,nbTimes)
W1.plotQuiver(proj='mill', res='i',instant=2, Dline=1)
#%%

# Plot the field using Basemap.  Start with setting the map
# projection using the limits of the lat/lon data itself:
fig = plt.figure()


m=Basemap(projection='mill',lat_ts=10,llcrnrlon=W.lon.min(), \
  urcrnrlon=W.lon.max(),llcrnrlat=W.lat.min(),urcrnrlat=W.lat.max(), \
  resolution='i')

x, y = m(*np.meshgrid(W.lon,W.lat))
velMag=np.empty(np.shape(W.u))

for t in range(np.size(W.time)) : 
    velMag[t]=(W.u[t]**2+W.v[0]**2)**0.5
m.pcolormesh(x,y,velMag[t],shading='flat',cmap=plt.cm.jet)
m.colorbar(location='right')
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(W.lat[0:-1:5],labels=[1,0,0,0])
m.drawmeridians(W.lon[0:-1:5],labels=[0,0,0,1])
plt.show()
#%%
fig = plt.figure()


m=Basemap(projection='mill',lat_ts=10,llcrnrlon=W.lon.min(), \
  urcrnrlon=W.lon.max(),llcrnrlat=W.lat.min(),urcrnrlat=W.lat.max(), \
  resolution='i')
          
x, y = m(*np.meshgrid(W.lon,W.lat))

plt.C=m.pcolormesh(x,y,W.v[0],shading='flat',cmap=plt.cm.jet)
plt.Q=m.quiver(x,y,W.u[0],W.v[0])

m.colorbar(location='right')
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(W.lat[0:-1:5],labels=[1,0,0,0])
m.drawmeridians(W.lon[0:-1:5],labels=[0,0,0,1])

def update_quiver(t,plt,u,v) :
    """method required to animate quiver and contour plot
    """
    plt.C=m.pcolormesh(x,y,W.wMag[t],shading='flat',cmap=plt.cm.jet)
    plt.Q=m.quiver(x,y,W.u[t],W.v[t])

    return plt

anim = animation.FuncAnimation(fig, update_quiver, frames=range(np.size(W.time)), fargs=(plt,W.u,W.v),
                               interval=50, blit=False)

plt.show()


#%%
# convert the lat/lon values to x/y projections.
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

X, Y = np.mgrid[:2*np.pi:0.2,:2*np.pi:0.2]
U = np.cos(X)
V = np.sin(Y)

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='r', units='inches')

ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = np.cos(X + num*0.1)
    V = np.sin(Y + num*0.1)

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=10, blit=False)

plt.show()

#%%
# plot the field using the fast pcolormesh routine 
# set the colormap to jet.



# Add a coastline and axis values.

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

# Add a colorbar and title, and then show the plot.

plt.title('Example 1: NWW3 Significant Wave Height from NOMADS')
plt.show()

#%%
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = './data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)


pathToSaveObj = './data/' + mydate + '_' + modelcycle + 'ens.obj'
Wspr = Weather.load(pathToSaveObj)
# we crop the Nan values
Wspr = Wspr.crop(timeSteps=[1, 64])

Tini = max([Wavg.time[0], Wspr.time[0]])
Wspr.time = Wspr.time - Tini
Wavg.time = Wavg.time - Tini

Wavg.getPolarVel()
#%%
anim=Wavg.animateQuiver()
anim.save('wind.mp4', fps=5, bitrate=2500, dpi=250)