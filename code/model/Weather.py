3#!/usr/bin/env python3
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
        This class is supposed to be used on GrAD's server files. No warranty however.
        class constructor,by default sets all attributes to None.
        lat, lon, time u and v must have same definition as in netCDF4 file of GrADS server.
        
        :ivar numpy.array lat: latitude in degree: array or list comprised in [-90 : 90]
                    
        :ivar numpy.array lon: longitude in degree : array or list comprised in [0 : 360]
           
        :ivar numy.array time: in days : array or list. Time is given in days (GrADS gives 81 times steps of 3hours so 
            it is 10.125 days with time steps of 0.125 days)
            
        :ivar u: velocity toward east in m/s. Must be of shape (ntime, nlat, nlon) 

        :ivar v: velocity toward north.
        
        :ivar scipy.interpolate.RegularGridInterpolator uInterpolator: \
            created when the method :py:meth:`Interpolators` is called
    
        :ivar scipy.interpolate.RegularGridInterpolator vInterpolator: \
            created when the method :py:meth:`Interpolators` is called
        
        :ivar numpy.array wMag: velocity magnitude  \
            created when the method :py:meth:`getPolarVel` is called
            
        :ivar numpy.array wang: velocity angle (polar coordinates)  \
            created when the method :py:meth:`getPolarVel` is called
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
        Takes a file path where a Weather object is saved and loads it into the script.
        
        If no lat or lon boundaries are defined, it takes the whole span present in the saved object.
        
        If no number of time step is defined it takes the whole span present if the saved object
        (but not more than 81 the value for GrAD files)
        
        :param string path: path to file of saved Weather object
        
        :param latBound: [minlat, maxlat], the largest span is [-90,90]
        :type latBound: list of int
        
        :param lonBound: [minlon, maxlon], the largest span is [0,360]
        :type lonBound: list of int
        
        :param int nbTimes: number of frames to load
        
        :return: loaded object
        :rtype: WeatherClass
            
        """
        filehandler = open(path, 'rb')
        obj = pickle.load(filehandler)
        filehandler.close()
        Cropped = obj.crop(latBound, lonBound, timeSteps)
        return Cropped
    
    @classmethod
    def download(cls, url, path, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 65]):
        """
        downloads Weather object from url server and writes it into path file.
        
        :param string url: url to server (designed for GrAD server)    
        
        :param other: same as :py:meth:`load` method.
        
        :return: the object corresponding to the downloaded weather
        :rtype: WeatherClass
        
        """

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
        Computes wind magnitude and direction and adds it to the object's attribute
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
        """
        Computes wind magnitude and direction from the velocities u and v
        
        :param float u: velocity toward east
        
        :param float v: velocity toward north
        
        :return: (magnitude, direction)
        :rtype: (float, float)
        """

        mag = (u ** 2 + v ** 2) ** 0.5
        ang = (180 / math.pi * math.atan2(u, v)) % 360
        return mag, ang

    def crop(self, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 81]):
        """
        Returns a cropped Weather object's data to the selected range of lon,lat and time steps.
        If no lat or lon boundaries are defined it takes the whole span present in the object.
        If no number of time step is defined it takes the whole span present if the object
        (but not more than 81 the value for GrAD files)

        :param latBound: [minlat, maxlat], the largest span is [-90,90]
        :type latBound: list of int
        
        :param lonBound: [minlon, maxlon], the largest span is [0,360]
        :type lonBound: list of int
        
        :param int nbTimes: number of frames to load
        
        :return: the cropped object
        :rtype: WeatherClass
        
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

    def plotQuiver(self, res='i', instant=0, Dline=5, density=1):
        """
        Plot the field (with Basemap) at one instant, using the limits of the lat/lon data
        Note: to plot whole earth, params should be close to: res='c',Dline=100,density=10
        
        :param int instant: time (idx) at which the data are plotted 
        :param string res: coast resolution: c (crude), l (low), i (intermediate), h (high), f (full) \
            http://matplotlib.org/basemap/api/basemap_api.html
        :param int Dline: sampling size for the lats and lons arrays (reduce dimensions)   
        :param int density: sampling size for the quiver array (reduce dimensions) 
           
        """
        
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        plt.figure()
        proj = 'mill'
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
    
    def plotMultipleQuiver(self, otherWeather, res='i', instant=0, Dline=5, density=1):
        """
        Same as plotQuiver but superimpose an other weather
        
        :param Weather otherWeather: the weather which is superimposed
        :param other: same as :py:meth:`plotQuiver` method. 
          
        """
        plt.figure()
        proj='mill'
        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))

        m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                 self.v[instant, 0::density, 0::density],color='black')
        # plot the other weather
        x, y = m(*np.meshgrid(otherWeather.lon, otherWeather.lat))
        m.quiver(x[0::density, 0::density], y[0::density, 0::density], otherWeather.u[instant, 0::density, 0::density],
                 otherWeather.v[instant, 0::density, 0::density], color = 'red')
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

    def plotColorQuiver(self, res='i', instant=0, Dline=5, density=1):
        """
        Same as plotQuiver but add the velocity magnitude in the background. \
            :py:meth:`getPolarVel` needs to be called before
        
        :param param: same as :py:meth:`plotQuiver` method. 
          
        """
        
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 22}

        matplotlib.rc('font', **font)
        plt.figure()
        proj='mill'
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

    def animateQuiver(self, res='i', instant=0, Dline=5, density=1, interval=50):
        """
        Animate a plotQuiver
        
        :param int interval: time between frames in ms
        :param other: same as :py:meth:`plotQuiver` method. 
        """
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        fig = plt.figure()
        proj='mill'
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
        
        """ 
        Add the u and v interpolators to the object (two new attributes :\
            uInterpolator and vInterpolator). Useful for future interpolations.
        
        """
        self.uInterpolator = rgi((self.time, self.lat, self.lon), self.u)
        self.vInterpolator = rgi((self.time, self.lat, self.lon), self.v)

