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
        This class is supposed to be used on GrAD's server files. No warranty however.
        class constructor,by default sets all attributes to None.
        lat, lon, time u and v must have same definition as in netCDF4 file of GrADS server.
        
        :ivar numpy.array lat: latitude in degree: array or list comprised in [-90 : 90]
                    
        :ivar numpy.array lon: longitude in degree : array or list comprised in [0 : 360]
           
        :ivar numy.array time: in days : array or list. Time is given in days (GrADS gives 81 times steps of 3hours so 
            it is 10.125 days with time steps of 0.125 days)
            
        :ivar u: velocity toward east in m/s. Must be of shape (ntime, nlat, nlon) 

        :ivar v: velocity toward north.

    """

    def __init__(self, lat=None, lon=None, time=None, u=None, v=None):
        """
        class constructor,by default sets all attributes to None.
        lat, lon, time u and v must have same definition as in netCDF4 file of GrADS server.
        """
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
        (but not more than 81 the value for GrAD files).
        
        :param str path: path to file of saved Weather object.
        
        :param latBound: [minlat, maxlat], lat span one wants to consider, the largest span is [-90,90].
        :type latBound: list of int.
        
        :param lonBound: [minlon, maxlon], lon span one wants to consider, the largest span is [0,360].
        :type lonBound: list of int.
        
        :param timeSteps: time steps of the forecasts one wants to load.
        :type timeSteps: list of int.
        
        :return: loaded object.
        :rtype: :any:`Weather`
            
        """
        filehandler = open(path, 'rb')
        obj = pickle.load(filehandler)
        filehandler.close()
        Cropped = obj.crop(latBound, lonBound, timeSteps)
        return Cropped
    
    @classmethod
    def download(cls, url, path, ens=False, latBound=[-90, 90], lonBound=[0, 360], timeSteps=[0, 65]):
        """
        Downloads Weather object from url server and writes it into path file.
        
        :param str url: url to server (designed for GrAD server).

        :param str path: path toward where the downloaded object is to be saved.

        :param bool ens: True is the downloaded data corresponds to a GEFS forecast, False for GFS.

        :param latBound: [minlat, maxlat], lat span one wants to consider, the largest span is [-90,90].
        :type latBound: list of int.

        :param lonBound: [minlon, maxlon], lon span one wants to consider, the largest span is [0,360].
        :type lonBound: list of int.

        :param timeSteps: time steps of the forecasts one wants to load.
        :type timeSteps: list of int.

        :return: the object corresponding to the downloaded weather.
        :rtype: :any:`Weather`

        """

        file = netCDF4.Dataset(url)
        lat = file.variables['lat'][:]
        lon = file.variables['lon'][:]
        # put time bounds !
        time = file.variables['time'][timeSteps[0]:timeSteps[1]]

        # latitude lower and upper index
        latli = np.argmin(np.abs(lat - latBound[0]))
        latui = np.argmin(np.abs(lat - latBound[1]))

        # longitude lower and upper index
        lonli = np.argmin(np.abs(lon - lonBound[0]))
        lonui = np.argmin(np.abs(lon - lonBound[1]))
        lat = lat[latli:latui]
        lon = lon[lonli:lonui]
        if ens : 
          u = file.variables['ugrd10m'][0,timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
          v = file.variables['vgrd10m'][0,timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]          
          
        else : 
          u = file.variables['ugrd10m'][timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]
          v = file.variables['vgrd10m'][timeSteps[0]:timeSteps[1], latli:latui, lonli:lonui]

        toBeSaved = cls(lat, lon, time, u, v)
        file.close()
        filehandler = open(path, 'wb')
        pickle.dump(toBeSaved, filehandler)
        filehandler.close()
        return toBeSaved

    def getPolarVel(self):
        """
        Computes wind magnitude and direction and adds it to the object's attribute as self.wMag (magnitude)
        and self.wAng (direction toward which the wind is blowing).
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
        Computes wind magnitude and direction from the velocities u and v.
        
        :param float u: velocity toward east.
        
        :param float v: velocity toward north.
        
        :return: magnitude, direction
        :rtype: float, float
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
        
        :return: the cropped object.
        :rtype: :any:`Weather`

        """

        if latBound != [-90, 90] or lonBound != [0, 360]:
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


        return Cropped

    def plotQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        Plots a quiver of the :any:`Weather` object's wind for a given instant. Basemap projection using the lat/lon limits of the data itself.


        :param str proj: `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_ projection method.
        :param str res: `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_ resolution.
        :param int instant: Time index at which the wind should be displayed.
        :param int Dline: Lat and lon steps to plot parallels and meridians
        :param int density: Lat and lon steps to plot quiver.

        :return: Plot framework.
        :rtype: `pyplot <https://matplotlib.org/api/pyplot_api.html>`_

        """

        # Start with setting the map.
        # projection using the limits of the lat/lon data itself:
        plt.figure()

        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))


        m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                 self.v[instant, 0::density, 0::density])
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(self.lat[0::Dline], labels=[1, 0, 0, 0])
        m.drawmeridians(self.lon[0::Dline], labels=[0, 0, 0, 1])

        plt.title('Wind amplitude and direction in [m/s] at time : ' + str(self.time[instant]) + ' days')
        plt.show()

        return plt
    
    def plotMultipleQuiver(self, otherWeather, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        Pretty much the same than :func:`plotQuiver` but to superimpose two quivers.

        :param Weather otherWeather: Second forecasts to be ploted with the one calling the method.
        """
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        plt.figure()

        m = Basemap(projection=proj, lat_ts=10, llcrnrlon=self.lon.min(), \
                    urcrnrlon=self.lon.max(), llcrnrlat=self.lat.min(), urcrnrlat=self.lat.max(), \
                    resolution=res)

        x, y = m(*np.meshgrid(self.lon, self.lat))

        m.quiver(x[0::density, 0::density], y[0::density, 0::density], self.u[instant, 0::density, 0::density],
                 self.v[instant, 0::density, 0::density],color='black')
        
        x, y = m(*np.meshgrid(otherWeather.lon, otherWeather.lat))
        m.quiver(x[0::density, 0::density], y[0::density, 0::density], otherWeather.u[instant, 0::density, 0::density],
                 otherWeather.v[instant, 0::density, 0::density], color = 'red')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(self.lat[0::Dline], labels=[1, 0, 0, 0])
        m.drawmeridians(self.lon[0::Dline], labels=[0, 0, 0, 1])
        plt.title('Wind amplitude and direction in [m/s] at time : ' + str(self.time[instant]) + ' days')
        plt.show()

        return plt

    def plotColorQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        Pretty much the same than :func:`plotQuiver` but on a contour plot of wind magnitude.
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

    def animateQuiver(self, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        Pretty much the same than :func:`plotQuiver` but animating the quiver over the different time steps
        starting at instant.
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
        """ Add the u and v interpolators to the object (two new attributes : 
            self.uInterpolator and self.vInterpolator).
            ::

                u = self.uInterpolator([t,lat,lon])
                #with u in m/s, t in days, lat and lon in degrees.
        """
        self.uInterpolator = rgi((self.time, self.lat, self.lon), self.u)
        self.vInterpolator = rgi((self.time, self.lat, self.lon), self.v)

