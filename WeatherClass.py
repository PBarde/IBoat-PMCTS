#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:11:08 2017

@author: paul
"""
import netCDF4
import pickle
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import RegularGridInterpolator as rgi


class Weather:
    """
    .. class::    Weather
    
        This class is supposed to be used on GrAD's server files. No warranty however.
        class constructor,by default sets all attributes to None.
        lat, lon, time u and v must have same definition as in netCDF4 file of GrADS server.
        
        * .. attribute :: lat : 
                    
            latitude in degree: array or list comprised in [-90 : 90]
                
        * .. attribute :: lon :
            
            longitude in degree : array or list comprised in [0 : 360]
            
        * .. attribute :: time :
            
            in days : array or list. Time is given in days (GrADS gives 81 times steps of 3hours so 
            it is 10.125 days with time steps of 0.125 days)
            
        * .. attribute :: u :
            
            velocity toward east in m/s. Must be of shape (ntime, nlat, nlon) 
            
        * .. attribute :: v :
            
            velocity toward north.
            
        * .. method :: load 
            
            loads a Weather object (see doc)
        
        * .. method :: download
            
            downloads data from server and writes it to Weather object (see doc)
            
        * .. method :: crop
            
            returns a cropped Weather object's data to the selected range of lon,lat and time steps (see doc)
            
        * .. method :: getPolarVel 
        
            computes wind magnitude and direction and adds it to the object's attribute
        
    """

    def __init__(self, lat=None, lon=None, time=None, u=None, v=None, wMag=None, wAng=None):
        self.lat = lat
        self.lon = lon
        self.time = time
        self.u = u
        self.v = v

    @classmethod
    def load(cls, path, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 81]):
        """
        .. method :: load

            **class method**, takes a file path where an Weather object is saved and loads it into the script.
            If no lat or lon boundaries are defined it takes the whole span present in the saved object.
            If no number of time step is defined it takes the whole span present if the saved object
            (but not more than 81 the value for GrAD files)

            * **path** - *string* : path to file of saved Weather object\n
            * **latBound** - *list of int* : [minlat, maxlat], the largest span is [-90,90]\n
            * **lonBound** - *list of int* : [minlon, maxlon], the largest span is [0,360]\n
            * **nbTimes** - *int* : number of frames to load
        """
        filehandler = open(path, 'rb')
        obj = pickle.load(filehandler)
        filehandler.close()
        Cropped = obj.crop(latBound, lonBound, timeSteps)
        return Cropped

    @classmethod
    def download(cls, url, path, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 81], ens=False):
        """
        .. method :: download

            **class method**, downloads Weather object from url server and writes it into path file.


            * **url** - *string* : url to server (designed for GrAD server)\n
            * **other params** : same as load method.

        """
        if ens == True and timeSteps == [0, 81]:
            timeSteps = [0, 65]

        file = netCDF4.Dataset(url)
        lat = file.variables['lat'][:]
        lon = file.variables['lon'][:]
        time = file.variables['time'][timeSteps[0]:timeSteps[1]]
        # put time bounds !
        #            lat_inds = np.where((lat > latBound[0]) & (lat < latBound[1]))
        #            lon_inds = np.where((lon > lonBound[0]) & (lon < lonBound[1]))
        #
        #            lat  = file.variables['lat'][lat_inds]
        #            lon  = file.variables['lon'][lon_inds]
        #            u = file.variables['ugrd10m'][time_inds,lat_inds,lon_inds]
        #            v = file.variables['vgrd10m'][time_inds,lat_inds,lon_inds]
        # latitude lower and upper index

        latli = np.argmin(np.abs(lat - latBound[0]))
        latui = np.argmin(np.abs(lat - latBound[1]))

        # longitude lower and upper index
        lonli = np.argmin(np.abs(lon - lonBound[0]))
        lonui = np.argmin(np.abs(lon - lonBound[1]))
        lat = lat[latli:latui]
        lon = lon[lonli:lonui]
        if ens == True:
            u = file.variables['ugrd10m'][0, timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
            v = file.variables['vgrd10m'][0, timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
        else:
            u = file.variables['ugrd10m'][timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
            v = file.variables['vgrd10m'][timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
        #
        #            u=file.variables['ugrd10m'][1,:,:]
        #            v=file.variables['vgrd10m'][1,:,:]
        toBeSaved = cls(lat, lon, time, u, v)
        file.close()
        filehandler = open(path, 'wb')
        pickle.dump(toBeSaved, filehandler)
        filehandler.close()
        return toBeSaved

    def getPolarVel(self):
        """
                 .. method :: getPolarVel 
        
            computes wind magnitude and direction and adds it to the object's attribute
        """
        self.wMag = np.empty(np.shape(self.u))
        self.wAng = np.empty(np.shape(self.u))
        for t in range(np.size(self.time)):
            self.wMag[t] = (self.u[t] ** 2 + self.v[t] ** 2) ** 0.5
            for i in range(np.size(self.lat)):
                for j in range(np.size(self.lon)):
                    self.wAng[t, i, j] = (180 / math.pi * math.atan2(self.u[t, i, j], self.v[t, i, j])) % 360

    @staticmethod
    def returnPolarVel(u, v):

        mag = (u ** 2 + v ** 2) ** 0.5
        ang = (180 / math.pi * math.atan2(u, v)) % 360
        return mag, ang

    def crop(self, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 81]):
        """
        .. method :: crop

            Returns a cropped Weather object's data to the selected range of lon,lat and time steps.
            If no lat or lon boundaries are defined it takes the whole span present in the object.
            If no number of time step is defined it takes the whole span present if the object
            (but not more than 81 the value for GrAD files)

            * **latBound** - *list of int* : [minlat, maxlat], the largest span is [-90,90]\n
            * **lonBound** - *list of int* : [minlon, maxlon], the largest span is [0,360]\n
            * **nbTimes** - *int* : number of frames to load
        """

        if (latBound != [-90, 90] or lonBound != [0, 360]):
            Cropped = Weather()
            lat_inds = np.where((self.lat > latBound[0]) & (self.lat < latBound[1]))
            lon_inds = np.where((self.lon > lonBound[0]) & (self.lon < lonBound[1]))
            Cropped.time = self.time[timeSteps[0]:timeSteps[1]]
            Cropped.lat = self.lat[lat_inds]
            Cropped.lon = self.lon[lon_inds]
            Cropped.u = np.empty((timeSteps[1] - timeSteps[0], np.size(lat_inds), np.size(lon_inds)))
            Cropped.v = np.empty((timeSteps[1] - timeSteps[0], np.size(lat_inds), np.size(lon_inds)))
            for time in range(timeSteps[1] - timeSteps[0]):
                i = 0
                for idlat in lat_inds[0]:
                    j = 0
                    for idlon in lon_inds[0]:
                        Cropped.u[time, i, j] = self.u[timeSteps[0] + time, idlat, idlon]
                        Cropped.v[time, i, j] = self.v[timeSteps[0] + time, idlat, idlon]
                        j = j + 1
                    i = i + 1



        elif latBound == [-90, 90] and lonBound == [0, 360] and timeSteps != [0, 81]:
            Cropped = Weather()
            Cropped.lat = self.lat
            Cropped.lon = self.lon
            Cropped.time = self.time[timeSteps[0]:timeSteps[1]]
            Cropped.u = self.u[timeSteps[0]:timeSteps[1]][:][:]
            Cropped.v = self.v[timeSteps[0]:timeSteps[1]][:][:]

        else:
            Cropped = self

        #            Cropped.getPolarVel()
        return Cropped

    def plotQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        to plot whole earth params should be close to res='c',Dline=100,density=10
        """
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        plt.figure()

        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))

        # m.pcolormesh(x,y,self.wMag[instant],shading='flat',cmap=plt.cm.jet)
        m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                 self.v[instant, 0::density, 0::density])
        # m.colorbar(location='right')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(self.lat[0::Dline], labels=[1, 0, 0, 0])
        m.drawmeridians(self.lon[0::Dline], labels=[0, 0, 0, 1])
        #        m.drawparallels(self.latGrid[0::Dline],labels=[1,0,0,0])
        #        m.drawmeridians(self.lonGrid[0::Dline],labels=[0,0,0,1])
        plt.title('Wind amplitude and direction in [m/s] at time : ' + str(self.time[instant]) + ' days')
        plt.show()

        return plt

    def plotColorQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        to plot whole earth params should be close to res='c',Dline=100,density=10
        """
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}

        matplotlib.rc('font', **font)
        plt.figure()

        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))

        m.pcolormesh(x, y, self.wMag[instant], shading='flat', cmap=plt.cm.jet)
        m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                 self.v[instant, 0::density, 0::density])
        cbar = m.colorbar(location='right')
        cbar.ax.set_ylabel('wind speed m/s')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(self.lat[0::Dline], labels=[1, 0, 0, 0])
        m.drawmeridians(self.lon[0::Dline], labels=[0, 0, 0, 1])
        plt.title('Wind amplitude and direction in [m/s] at time : ' + str(self.time[instant]) + ' days')
        plt.show()

        return plt

    def animateQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1, interval=50):
        """
        to plot whole earth params should be close to res='c',Dline=100,density=10
        """
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        fig = plt.figure()

        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))

        plt.C = m.pcolormesh(x, y, self.wMag[instant], shading='flat', cmap=plt.cm.jet)
        plt.Q = m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                         self.v[instant, 0::density, 0::density])
        m.colorbar(location='right')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(self.lat[0::Dline], labels=[1, 0, 0, 0])
        m.drawmeridians(self.lon[0::Dline], labels=[0, 0, 0, 1])

        def update_quiver(t, plt, self):
            """method required to animate quiver and contour plot
            """
            plt.C = m.pcolormesh(x, y, self.wMag[instant + t], shading='flat', cmap=plt.cm.jet)
            plt.Q = m.quiver(x[0::density, 0::density * 2], y[0::density, 0::density * 2],
                             self.u[instant + t, 0::density, 0::density * 2],
                             self.v[instant + t, 0::density, 0::density * 2])
            plt.title('Wind amplitude and direction in [m/s] at time : ' + str(self.time[instant + t]) + ' days')
            return plt

        anim = animation.FuncAnimation(fig, update_quiver, frames=range(np.size(self.time[instant:])),
                                       fargs=(plt, self),
                                       interval=50, blit=False)

        plt.show()

        return anim

    def Interpolators(self):
        self.uInterpolator = rgi((self.time, self.lat, self.lon), self.u)
        self.vInterpolator = rgi((self.time, self.lat, self.lon), self.v)

