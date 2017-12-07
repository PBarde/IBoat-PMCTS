#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

:Autors: Paul Barde & Fabien Brulport

Module encapsulating all the classes required to run a simulation. 

"""

import numpy as np
import math
import random as rand
from WeatherTLKT import Weather
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import animation
from math import sin,cos,asin,atan2,acos,pi
from math import radians as rad


#: Actions that are authorized i.e. headings the boat can follow. Must be sorted.
ACTIONS = tuple(np.arange(0,360,45))

#: Constant to convert days in seconds.
DAY_TO_SEC=24*60*60

#: Earth radius in meters.
EARTH_RADIUS = 6371e3

#: Angular margin that characterizes the destination point.
DESTINATION_ANGLE=rad(0.005)
                        
class Simulator :
    """
    Class emboding the boat and weather interactions with also the tools required\
    to do projection on earth surface. For now, only used by the MCTS tree search.

    :ivar numpy.array times: Vector of the instants used for the simulation \
        in days.

    :ivar numpy array lons: Longitudes in degree in [0 , 360].

    :ivar numpy.array lats: Latitudes in degree in [-90 : 90].

    :ivar list state: Current state [time index, lat, lon] of the boat in \
        (int,degree,degree).

    :ivar list prevState: Previous state [time index, lat, lon] of the boat in \
        (int,degree,degree).

    :ivar uWindAvg: Interpolator for the wind velocity blowing toward West. Generated at initialisation with :py:meth:`WeatherTLKT.Weather.Interpolators`.
    :vartype uWindAvg: `Interpolator`_

    :ivar vWindAvg: Interpolator for the wind velocity blowing toward North. Generated at initialisation with :py:meth:`WeatherTLKT.Weather.Interpolators`.
    :vartype vWindAvg: `Interpolator`_

    """
  
    def __init__(self,times,lats,lons,WeatherAvg,stateInit) : 
        """
        Class constructor
        """
      
        self.times=times
        self.lats=lats
        self.lons=lons
        self.state=list(stateInit)
        self.prevState=list(stateInit)
  
        WeatherAvg.Interpolators()
        self.uWindAvg=WeatherAvg.uInterpolator
        self.vWindAvg=WeatherAvg.vInterpolator
     
    def reset(self,stateInit):
        """
        Reset the simulated boat to a specific state.
        
        :param list stateInit: State to which the simulator is reinitialized.
        """
        
        self.state = list(stateInit)
        self.prevState = list(stateInit)
        
    
    def getDistAndBearing(self,position,destination):
        """
        Returns the distance and the initial bearing to follow to go to\
        a destination following a great circle trajectory (orthodrome). `Link to documentation`_
        
        :param position: Current position of the boat.
        :type position: list(float : lat, float : lon)
           
        :param destination: Point toward which the distance and initial bearing\
            are computed.
        :type destination: list(float : lat, float : lon)
           
          
        :return: Shortest distance between the two points in meters, and \
            initial bearing of the orthodrome trajectory in degrees. 
        :rtype: float: distance, float: bearing
        """
        latDest,lonDest = [rad(destination[0]), rad(destination[1])]
        latPos, lonPos = [rad(position[0]), rad(position[1])]
  
        a=(sin((latDest-latPos)/2))**2+cos(latPos)*cos(latDest)*(sin((lonDest-lonPos)/2))**2
        
        distance=2*EARTH_RADIUS*atan2(a**0.5,(1-a)**0.5)
        
        x=math.cos(latDest)*math.sin(lonDest-lonPos)
        y=math.cos(latPos)*math.sin(latDest)-math.sin(latPos)*math.cos(latDest)\
                  *math.cos(lonDest-lonPos)
                  
        bearing=(math.atan2(x,y)*180/math.pi+360)%360
      
        return distance,bearing
  
  
    def getDestination(self,distance,bearing,departure):
        """
        Returns the destination point following a orthodrome trajectory for a\ 
        given bearing and distance. `Link to
        documentation <http://www.movable-type.co.uk/scripts/latlong.html>`_.
        
        :param float distance: Distance in meters to the destination. 

        :param float bearing: Initial bearing of the orthodrome trajectory\
            starting at departure and ending at destination. In degrees.
        
        :param departure: Departure point of the trajectory.
        :type departure: list(float : lat, float : lon)
           
          
        :return: Destination reached following the othodrome trajectory. 
        :rtype: [float : lat, float : lon]
          
        """
        
        latDep, lonDep = [rad(departure[0]), rad(departure[1])]
        
        bearing=rad(bearing)
        
        latDest=asin(sin(latDep)*cos(distance/EARTH_RADIUS) + \
                         cos(latDep)*sin(distance/EARTH_RADIUS)*cos(bearing))
        
        lonDest=lonDep + atan2(sin(bearing)*sin(distance/EARTH_RADIUS) \
                                    *cos(latDep),cos(distance/EARTH_RADIUS)\
                                             -sin(latDep)*sin(latDest))
        
        latDest=(latDest*180/math.pi)
        lonDest=(lonDest*180/math.pi)
                          
        return [latDest, lonDest]
    
    def getWind(self) :
        """
        Returns the wind at the current simulator state.
        
        :return: Wind toward the East in m/s and wind toward the North in m/s.
        :rtype: float : uAvg, float : vAvg

        """
        uAvg=self.uWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
        vAvg=self.vWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
        
        return uAvg,vAvg
    
    
    def doStep(self,action):
        """
        Does one iteration of the simulation (one time step) following a provided action.\
        Updates and returns the new boat state. Supposes constant wind during a time step. Also\
        constant boat speed. Assumes that the boat moves following orthodomes trajectories. This \
        is correct if the distances covered during the time step is relatively small (and we\
        are not close to the poles): the orthodrome's headings do not vary much. 
        
        :param float action: Boat's heading in degree.
        
        :return: Reference toward the updated simulator's state.
        :rtype: self.state
        """
        
        #we copy the current state into the previous one
        self.prevState=list(self.state)
        
        #we get the wind at current state
        uWind,vWind=self.getWind()
        windMag,windAng=Weather.returnPolarVel(uWind,vWind)
        
        #We get speed from wind on sail
        pOfSail=abs((windAng+180)%360-action)
        boatSpeedDet=Boat.getDeterDyn(pOfSail,windMag,Boat.FIT_VELOCITY)
        boatSpeed = Boat.addUncertainty(boatSpeedDet)
        
        # We integrate it
        Dt=(self.times[self.state[0]+1]-self.times[self.state[0]])*DAY_TO_SEC
        DL=boatSpeed*Dt # distance travelled
        
        #new position, correct if the boat follows an orthodrome
        newLat,newLon=self.getDestination(DL,action,[self.state[1],self.state[2]])
        
        self.state[0],self.state[1],self.state[2]=self.state[0]+1,newLat,newLon
        
        return self.state
    
    
    @staticmethod
    def fromGeoToCartesian(coordinates) : 
      """
      Transforms geographic coordinates to cartesian coordinates. The cartesian frame \
      has its origin at the center of the earth. Its orientation and so on is not explicitely given\
      since the function is only used to create a plane. 
      
      :param coordinates: Coordinates in geographical frame.
      :type coordinates: [float : lat, float : lon]
        
      :return: x,y,z coordinates.
      :rtype: [float, float, float]
         
      """
      
      lat,lon=coordinates[:]
      lat,lon=rad(lat),rad(lon)
      x=sin(lat)*cos(lon)
      y=sin(lat)*sin(lon)
      z=cos(lat)
      return [x,y,z]
      
    
    def prepareBaseMap(self,proj='mill',res='i',Dline=5,dl=1.5,dh=1,centerOfMap=None) :
        """
        Prepares the figure to plot a trajectory. Based on mpl_toolkits.basemap.Basemap.
        
        :param str proj: Name of the projection (default Miller) (see `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_ doc).
        
        :param str res: Resolution (see `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_ doc)
        
        :param int Dline: sampling size for the lats and lons arrays (reduce dimensions)            
        
        :return: The initialized basemap.
        :rtype: `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_
        """
        if proj == 'mill' : 
          plt.figure()
          basemap=Basemap(projection=proj,llcrnrlon=self.lons.min(), \
            urcrnrlon=self.lons.max(),llcrnrlat=self.lats.min(),urcrnrlat=self.lats.max(), \
            resolution=res)
          basemap.drawcoastlines()
          basemap.fillcontinents()
          basemap.drawmapboundary()
          basemap.drawparallels(self.lats[0::Dline],labels=[1,0,0,0])
          basemap.drawmeridians(self.lons[0::2*Dline],labels=[0,0,0,1])
          
        elif proj=='aeqd' : 
          plt.figure()
          wdth = (self.lons[-1]-self.lons[0])*dh*math.pi/180*EARTH_RADIUS
          hght = (self.lats[-1]-self.lats[0])*dl*math.pi/180*EARTH_RADIUS
          basemap = Basemap(width=wdth,height=hght,projection='aeqd',lat_0=centerOfMap[0],lon_0=centerOfMap[1],resolution=res)
          basemap.drawcoastlines()
          basemap.fillcontinents()
          basemap.drawmapboundary()
          basemap.drawparallels(self.lats[0::Dline],labels=[1,0,0,0])
          basemap.drawmeridians(self.lons[0::2*Dline],labels=[0,0,0,1])
          
        return basemap
    
    def plotTraj(self,states,basemap,quiv=False,scatter=False,color='black'):
        """
        Draw the states on the map either as a trajectory and/or scatter of points. Can also plot mean wind
        for each state and return it.
        
        :param states: List of all the state (state is an array)
        :type states: array or list
        
        :param basemap: Basemap object on which the trajectory will be drawn
        :type basemap: `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_
        :param boolean quiv: If True, shows the wind at each time step and returns it.
        
        :param boolean scatter: If True, plots the positions of the boat at\
            each time step as scatter only
        
        :param str color: Color of the trajectory points

        :return: u,v if quiver is true
        :rtype: float,float

        """
        states=np.array(states)
        posLat=states[:,1]
        posLon=states[:,2]
        times=self.times[states[:,0].astype(int)]
        x,y=basemap(posLon,posLat)
        
        if scatter :
            basemap.scatter(x,y,zorder=0,c=color,s=100)
            
        else:
            basemap.plot(x,y,markersize=4,zorder=0,color=color)
            basemap.scatter(x[-1],y[-1],zorder=1,color=color)
            
        
        if quiv : 
            points=zip(times,posLat,posLon)
            u=self.uWindAvg(list(points))
            points=zip(times,posLat,posLon)
            v=self.vWindAvg(list(points))
            
            print('u= ' + str(u) + '\n')
            print('v= ' + str(v) + '\n')
            basemap.quiver(x,y,u,v,zorder=2,width=0.004,color='teal')
            
        if quiv : 
            return u,v
        
    def animateTraj(self,windAvg, states, trajSteps=3, proj='mill', res='i', instant=0, Dline=5, density=1):
        """
        Animates the trajectory corresponding to the list of states.

        :param Weather windAvg: The weather object corresponding to the trajectory.
        :param list states: List of boat states along its trajectory.
        :param int trajSteps: State step for the animation (a plot update corresponds to trajSteps covered states)
        :param str proj: Projection to be used. Refer to `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_
        :param str res: Plot resolution. Refer to `Basemap <https://matplotlib.org/basemap/api/basemap_api.html#module-mpl_toolkits.basemap>`_
        :param int instant: Initial instant of the animation.
        :param int Dline: Lat and lon steps to plot parallels and meridians.
        :param int density: Lat and lon steps to plot quiver.
        :return: Animation function.
        :rtype: `FuncAnimation <https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_

        """
        # to plot whole earth params should be close to res='c',Dline=100,density=10
        # Plot the field using Basemap.  Start with setting the map
        # projection using the limits of the lat/lon data itself:
        fig=plt.figure()
        m=Basemap(projection=proj,lat_ts=10,llcrnrlon=windAvg.lon.min(), \
          urcrnrlon=windAvg.lon.max(),llcrnrlat=windAvg.lat.min(),urcrnrlat=windAvg.lat.max(), \
          resolution=res)
        
        xquiv, yquiv = m(*np.meshgrid(windAvg.lon,windAvg.lat))
        
        Q=m.quiver(xquiv[0::density,0::density],yquiv[0::density,0::density],windAvg.u[instant,0::density,0::density],windAvg.v[instant,0::density,0::density])
        
        states=np.array(states)
        posLat=states[:,1]
        posLon=states[:,2]
        x,y=m(posLon,posLat)
        
        T=m.plot(x[0:instant*3],y[0:instant*3],linewidth=4)[0]
        
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(windAvg.lat[0::Dline],labels=[1,0,0,0])
        m.drawmeridians(windAvg.lon[0::2*Dline],labels=[0,0,0,1])
        
        def update_quiver(t,Q,T,windAvg) :
            """method required to animate quiver and contour plot
            A FAIRE
            """
            Q.set_UVC(windAvg.u[instant+t,0::density,0::density*2],windAvg.v[instant+t,0::density,0::density*2])
            T.set_data(x[0:instant+t*3],y[0:instant+t*3])
            plt.title('time : ' + str(windAvg.time[instant+t]-windAvg.time[0]) + ' days')       
            return plt
        
        anim = animation.FuncAnimation(fig, update_quiver, frames=range(int(len(states)/trajSteps)),fargs=(Q,T,windAvg))
  
        plt.show()
  
  
        return anim

class Boat : 
    """
  Class defining the boat's dynamics.
    
    """

    #:Coefficients of the polar fitted with a 5th order two dimensionnal polynomial.
    FIT_VELOCITY=((-0.0310198207067136,-0.0600881995286002,0.0286695485969272, -0.00684406929296715, 0.000379636836557256, -6.77704610076153e-06), \
                (0.0106968590293653, 0.00665508747173206, -4.03686836415123e-05, 2.38962919033178e-05, -6.16724919464073e-07,0), \
                (-0.000370260554113750, -7.68003222256045e-05, -2.44425618051996e-06, -2.16961875441841e-08,0,0),\
                (4.94175336099537e-06, 5.68757177397705e-07, 1.17646387245609e-08,0,0,0),\
                (-2.91900968067076e-08, -1.67287818401469e-09,0,0,0,0),\
                (6.34463723894578e-11,0,0,0,0,0))

    #: Maximal wind magnitude acceptable by the polar fit. If experienced magnitude is greater the wind magnitude is
    #: set to POLAR_MAX_WIND_MAG.
    POLAR_MAX_WIND_MAG = 19.1
    #: Minimal point of sail where the boat can sail without tacking.
    POLAR_MIN_POFSAIL = 35
    #: Maximal point of sail where the boat can sail without tacking.
    POLAR_MAX_POFSAIL = 160
    #: Characterizes the uncertainty on the boat's dynamics.
    UNCERTAINTY_COEFF = 0.2

    @staticmethod
    def getDeterDyn(pOfSail,windMag,fitCoeffs) :
        """
        Returns the deterministic boat velocity for a given wind magnitude and point of sail.

        :param float pOfSail: point of sail of the boat (in degree).
        :param float windMag: magnitude of the winf in m/s.
        :param tuple fitCoeffs: coefficients of the fitted velocity polar cf: :any:`FIT_VELOCITY`.
        :return: deterministic boat's speed in m/s (in direction of heading).
        :rtype: float
        """

        if pOfSail > 180 :
          pOfSail = 360-pOfSail

        if windMag > Boat.POLAR_MAX_WIND_MAG :
          windMag=Boat.POLAR_MAX_WIND_MAG

        if pOfSail < Boat.POLAR_MIN_POFSAIL :
          speedAtMinPofSail=math.cos(math.pi*Boat.POLAR_MIN_POFSAIL/180)* \
                                    np.polynomial.polynomial.polyval2d(Boat.POLAR_MIN_POFSAIL,windMag,fitCoeffs)
          return speedAtMinPofSail/(math.cos(math.pi*pOfSail/180))

        elif pOfSail>Boat.POLAR_MAX_POFSAIL :
          speedAtMaxPofSail=math.cos(math.pi*Boat.POLAR_MAX_POFSAIL/180)* \
                                    np.polynomial.polynomial.polyval2d(Boat.POLAR_MAX_POFSAIL,windMag,fitCoeffs)
          return speedAtMaxPofSail/(math.cos(math.pi*pOfSail/180))

        else :
          return np.polynomial.polynomial.polyval2d(pOfSail,windMag,fitCoeffs)

    @staticmethod
    def addUncertainty(boatSpeed):

        """
        Returns the noisy boat velocity.

        :param float boatSpeed: deterministic boat velocity in m/s.
        :return: Stochastic boat speed in m/s.
        :rtype: float

        """
        boatSpeedNoisy=rand.gauss(boatSpeed,boatSpeed*Boat.UNCERTAINTY_COEFF)
  
        return boatSpeedNoisy


                
                
                
                
                        
                
                
                
                
                
                
                
            
            
            

                
                
