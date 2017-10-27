#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:24:21 2017

@author: paul

Module encapsulating all the classes required to run a simulation. 

Constants
---------

ACTIONS : np.array() : 
  Actions that are autorized i.e. headings the boat can follow. 
  
DAY_TO_SEC 
  _
  

EARTH_RADIUS 
  _
  
DESTINATION_ANGLE : float : 
  Angle in radians defining the destination's zone. 


  
"""
import numpy as np
import math
import random as rand
from WeatherClass import Weather
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import animation
from math import sin,cos,asin,atan2,acos,pi
from math import radians as rad


"""MUST BE SORTED"""
ACTIONS = np.arange(0,360,45) 
ACTIONS=tuple(ACTIONS)
DAY_TO_SEC=24*60*60
EARTH_RADIUS = 6371e3
DESTINATION_ANGLE=rad(0.005)
                        
class Simulator :
    """
    Class emboding the boat and weather interactions with also the tools required\
    to do projection on earth surface. For now, only used by the MCTS tree search. 
  
  
    Attributes
    ----------
      
    times : numpy array :
        Vector of the instants used for the simulation in days. 
    
    lons : numpy array :
        Longitudes in degree in [0 , 360].
          
    lats : numpy array :
        Latitudes in degree in [-90 : 90]. 
          
    state : list :
        Current state [time, lat, lon] of the boat in (days,degree,degree)
  
    prevState : list :
        Previous state [time, lat, lon] of the boat in (days,degree,degree)
      
    uWindAvg : scipy.interpolate.interpolate.RegularGridInterpolator :
        Interpolator for the wind velocity blowing toward West.\
        Generated at initialisation. 
        
    vWindAvg : scipy.interpolate.interpolate.RegularGridInterpolator :
        Interpolator for the wind velocity blowing toward North.
        Generated at initialisation. 
        
    Methods
    ---------
    
    reset : 
      Resets the boat to a specific state.
          
    getDistAndBearing : 
      Returns the distance and the initial bearing to follow to go to\
      a destination following a great circle trajectory (orthodrome). Link to\
      documentation http://www.movable-type.co.uk/scripts/latlong.html.
          
    getDestination :
      Returns the destination point following a orthodrome trajectory for a given\
      bearing and distance. Link to\
      documentation http://www.movable-type.co.uk/scripts/latlong.html.
          
    getWind :
      Returns the wind at the current simulator state.
      
    doStep : 
      Does one iteration of the simulation (one time step) following a provided action.\
      Updates and returns the new boat state.
      
    fromGeoToCartesian :
      Transforms geographic coordinates to cartesian coordinates. 
          
    prepareBaseMap :
      Prepares a figure and a BaseMap projection to plot a trajectory. Based on mpl_toolkits.basemap.
      
    plotTraj :
      Projects and plot a trajectory.
      
    animateTraj : 
      Animates a trajectory on a map with the corresponding wind. 
    """
  
    def __init__(self,times,lats,lons,WeatherAvg,stateInit) : 
        """
        Class constructor.
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
        
        Parameters
        ----------
        state : list : 
          State to which the simulator is reinitialized. 
        """
        
        self.state=list(stateInit)
        self.prevState=list(stateInit)
        
    
    def getDistAndBearing(self,position,destination):
        """
        Returns the distance and the initial bearing to follow to go to\
        a destination following a great circle trajectory (orthodrome). Link to\
        documentation http://www.movable-type.co.uk/scripts/latlong.html.
        
        Parameters
        ----------
        position : list(float : lat, float : lon) :
          Current position of the boat. 
        
        destination : list(float : lat, float : lon) :
          Point toward which the distance and initial bearing are computed. 
          
        Return
        ------
        distance : float :
          Shortest distance between the two points in meters. 
        
        bearing : float :
          Initial bearing of the orthodrome trajectory in degrees. 
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
        Returns the destination point following a orthodrome trajectory for a given\
        bearing and distance. Link to\
        documentation http://www.movable-type.co.uk/scripts/latlong.html.
        
        Parameters
        ----------
        distance : float : 
          Distance in meters to the destination. 
          
        bearing : float : 
          Initial bearing of the orthodrome trajectory starting at departure and ending at destination.\
          In degrees. 
          
        departure : list(float : lat, float : lon) :
          Departure point of the trajectory. 
          
        Returns
        -------
        destination : list(float : lat, float : lon) :
          Destination reached following the othodrome trajectory. 
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
        
        Returns : 
        ---------
        
        uAvg : float : 
          Wind toward the East in m/s. 

        vAvg : float : 
          Wind toward the North in m/s.           
        
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
        
        Parameters
        ----------
        
        action : float : 
          Boat's heading in degree. 
        
        Returns
        -------
        
        self.state : ref : 
          Reference toward the updated simulator's state. 
        
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
      
      Parameters
      ----------
      
      coordinates : list(float : lat, float : lon) :
        Coordinates in geographical frame.
        
      Returns
      -------
      
      coordinates : list(float, float, float) : 
        x,y,z coordinates. 
      """
      
      lat,lon=coordinates[:]
      lat,lon=rad(lat),rad(lon)
      x=sin(lat)*cos(lon)
      y=sin(lat)*sin(lon)
      z=cos(lat)
      return [x,y,z]
      
    
    def prepareBaseMap(self,proj='mill',res='i',Dline=5,dl=1.5,dh=1,centerOfMap=None) :
        """
        Prepares the figure to plot a trajectory. Based on basemap.
        
        Parameters
        ----------
        proj : str:
            Name of the projection (default Miller)
        res : str:
            Resolution (see Basemap doc)
        Dline : int:
            sampling size for the lats and lons arrays (reduce dimensions) 
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
    
    def plotTraj(self,states,basemap,quiv=False,heading=225,scatter=False,color='black'):
        """
        Draw the states on the map
        
        Parameters
        ----------
        states : array or list:
            List of all the state (state is an array)
        map : Basemap:
            Basemap object on which the trajectory will be drawn
        quiv : boolean:
            If True, shows the wind at each time step
        heading : int:
            PAS UTILISEE ??
        scatter : boolean:
            If True, plots the positions of the beat at each time step as 
            scatter only
        color : str:
            Color of the trajectory points
        
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
        
    def animateTraj(self,windAvg, states, trajSteps=3, proj='mill', res='i', instant=0, Dline=5, density=1, interval=1): 
        """
        A FAIRE
        Animate the Trajectory
        
        
        to plot whole earth params should be close to res='c',Dline=100,density=10
        """
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
  #                S=m.scatter(x[instant*trajSteps],y[instant*trajSteps])[0]
        
        m.drawcoastlines()
        m.fillcontinents()
        m.drawmapboundary()
        m.drawparallels(windAvg.lat[0::Dline],labels=[1,0,0,0])
        m.drawmeridians(windAvg.lon[0::2*Dline],labels=[0,0,0,1])
        
        def update_quiver(t,Q,T,windAvg) :
            """method required to animate quiver and contour plot
            """
            Q.set_UVC(windAvg.u[instant+t,0::density,0::density*2],windAvg.v[instant+t,0::density,0::density*2])
            T.set_data(x[0:instant+t*3],y[0:instant+t*3])
            plt.title('time : ' + str(windAvg.time[instant+t]-windAvg.time[0]) + ' days')       
            return plt
        
        anim = animation.FuncAnimation(fig, update_quiver, frames=range(int(len(states)/3)),fargs=(Q,T,windAvg))
  
        plt.show()
  
  
        return anim

class Boat : 
    """
  Class defining the boat's dynamics. 
  
  Constants
  ---------
    FIT_VELOCITY : tuple with shape (6,6)
      Coefficients of the polar fitted with a 5th order two dimensionnal polynomial.
    
    POLAR_MAX_WIND_MAG : int
      Maximal wind magnitude acceptable by the polar fit. If experienced  magnitude is greater,\
      the wind magnitude is set to POLAR_MAX_WIND_MAG. 
    
    POLAR_MIN_POFSAIL : int
      Minimal point of sail where the boat can sail without tacking.
    
    POLAR_MAX_POFSAIL : int
      Maximal point of sail where the boat can sail without tacking.
      
    UNCERTAINTY_COEFF : int
      Caracterizes the uncertainty on the boat's dynamics. 
      
  Methods
  -------
    getDeterDyn : 
      Returns the deterministic boat velocity for a given wind magnitude and point of sail. 
      
    addUncertainty : 
      Returns the noisy boat velocity.
    
    """
       # boat dynamic parameters
    FIT_VELOCITY=((-0.0310198207067136,-0.0600881995286002,0.0286695485969272, -0.00684406929296715, 0.000379636836557256, -6.77704610076153e-06), \
                (0.0106968590293653, 0.00665508747173206, -4.03686836415123e-05, 2.38962919033178e-05, -6.16724919464073e-07,0), \
                (-0.000370260554113750, -7.68003222256045e-05, -2.44425618051996e-06, -2.16961875441841e-08,0,0),\
                (4.94175336099537e-06, 5.68757177397705e-07, 1.17646387245609e-08,0,0,0),\
                (-2.91900968067076e-08, -1.67287818401469e-09,0,0,0,0),\
                (6.34463723894578e-11,0,0,0,0,0))
    POLAR_MAX_WIND_MAG = 19.1
    POLAR_MIN_POFSAIL = 35
    POLAR_MAX_POFSAIL = 160
    UNCERTAINTY_COEFF = 0.2
      
    @staticmethod
    def getDeterDyn(pOfSail,windMag,fitCoeffs) :
      """ bla bla"""
        
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
        
      boatSpeedNoisy=rand.gauss(boatSpeed,boatSpeed*Boat.UNCERTAINTY_COEFF)
  
      return boatSpeedNoisy


                
                
                
                
                        
                
                
                
                
                
                
                
            
            
            

                
                