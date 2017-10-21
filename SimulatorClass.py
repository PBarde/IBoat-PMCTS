#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:24:21 2017

@author: paul
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
from bisect import bisect_left

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
BOAT_UNCERTAINTY_COEFF = 0.2
#BOAT_UNCERTAINTY_COEFF = 0
#BOAT_MAX_SPEED = 2.5 #m/s
#BOAT_UNCERTAINTY_COEFF = 0.

"""MUST BE SORTED"""
ACTIONS = np.arange(0,360,45) 
ACTIONS=tuple(ACTIONS)
ACTIONS_DP=tuple(np.arange(0,360,1))


DAY_TO_SEC=24*60*60
EARTH_RADIUS = 6371e3
DESTINATION_ANGLE=rad(0.005)
                        
class Simulator :
        """
    .. class::    Simulator
    
        Description of the class
        
        * .. attribute :: times : 
                    
            time of the simulation in days: array
                
        * .. attribute :: lons :
            
            longitudes in degree : array or list comprised in [0 : 360]. 
            
        * .. attribute :: lats :
            
            latitudes in degree : array or list comprised in [-90 : 90]. 
            
        * .. attribute :: state :
            
            current state of the boat : array [time, lat, lon]
        
        * .. attribute :: prevState :
            
            previous state of the boat : array [time, lat, lon]
            
        * .. attribute :: uWindAvg :
            
            east wind velocity interpolator

        * .. attribute :: vWindAvg :
            
            north wind velocity interpolator    
            
        * .. method :: reset(stateInit = [time, lat, lon])
            
            reset the boat to a specific state
        
        * .. method :: getDistAndBearing(position = [lat, lon], destination = [lat, lon])
            
            returns the distance and the bearing to get from 'position' to 'destination'
            
        * .. method :: getDestination(distance, bearing, departure = [lat, lon])
        
            returns the destination point, when the departure, the distance and
            the bearing is given. 
            
        * .. method :: getWind 
        
            returns the wind [uAvg, vAvg] at the actual state (time and position)
        
        * .. method :: doStep(action) 
        
            Does one iteration of the simulation (one time step) taking into
            account the given 'action'.  Updates and returns the new boat state.
            
        * .. method :: preparePlotTraj
        
            prepares the figure to plot a trajectory. Based on basemap.
            Default: Miller Projection
            
        * .. method :: preparePlotTraj2(action) 
        
            prepares the figure to plot a trajectory. Based on basemap.
            Default: Miller Projection
    """
    
        def __init__(self,times,lats,lons,WeatherAvg,stateInit) :
            
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
            reset the boat to a specific state
            """
            
            self.state=list(stateInit)
            self.prevState=list(stateInit)
            
        
        def getDistAndBearing(self,position,destination):
            """
            return the distance and the bearing to from position to destination
            """
            latDest,lonDest = [rad(destination[0]), rad(destination[1])]
            latPos, lonPos = [rad(position[0]), rad(position[1])]

            a=(sin((latDest-latPos)/2))**2+cos(latPos)*cos(latDest)*(sin((lonDest-lonPos)/2))**2
            
            distance=2*EARTH_RADIUS*atan2(a**0.5,(1-a)**0.5)
            
            x=math.cos(latDest)*math.sin(lonDest-lonPos)
            y=math.cos(latPos)*math.sin(latDest)-math.sin(latPos)*math.cos(latDest)\
                      *math.cos(lonDest-lonPos)
                      
            bearing=(math.atan2(x,y)*180/math.pi+360)%360
            
#                print(str(bearing))
            return distance,bearing


        def getDestination(self,distance,bearing,departure):
            """
            returns the destination point, when the departure, the distance and
            the bearing is given. 
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
#                print(str(bearing))
                              
            return [latDest, lonDest]
        
        def getWind(self) :
            """
            returns the wind at the actual state (time and position)
            """
            uAvg=self.uWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
            vAvg=self.vWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
            
            return uAvg,vAvg
        
        
        def doStep(self,action):
            """
            Does one iteration of the simulation (one time step) taking into
            account the given 'action'.  Updates and returns the new boat state.
            ???
            CAREFULLL : ONLY VALID FOR SMALL TIME STEPS (1HOUR) 
            else Dlat and Dlon might be wrong and we need to use the getPolarVel 
            and getDestination method"""
            
            # A ENLEVER ???
#                action=Simulator.takeClosest(ACTIONS_DP,action)
#                print(action)
            
            #we copy the current state into the previous one
            self.prevState=list(self.state)
            
            #we get the wind at current state
            uWind,vWind=self.getWind()
            windMag,windAng=Weather.returnPolarVel(uWind,vWind)
            
            #We get speed from wind on sail
            pOfSail=abs((windAng+180)%360-action)
            boatSpeedDet=Boat.getDeterDyn(pOfSail,windMag,FIT_VELOCITY)
            boatSpeed = Boat.addUncertainty(boatSpeedDet)
            
            # We integrate it
            Dt=(self.times[self.state[0]+1]-self.times[self.state[0]])*DAY_TO_SEC
            """ This is correct for big timeSteps"""
            DL=boatSpeed*Dt # distance travelled
            
            #new position
            newLat,newLon=self.getDestination(DL,action,[self.state[1],self.state[2]])
            
            self.state[0],self.state[1],self.state[2]=self.state[0]+1,newLat,newLon
            
            return self.state
            
        @staticmethod
        def takeClosest(myList, myNumber):
            """
            Assumes myList is sorted. Returns closest value to myNumber.
        
            If two numbers are equally close, return the smallest number.
            """
            pos = bisect_left(myList, myNumber)
            if pos == 0:
                return myList[0]
            if pos == len(myList):
                return myList[-1]
            before = myList[pos - 1]
            after = myList[pos]
            if after - myNumber < myNumber - before:
               return after
            else:
               return before
        
        
        @staticmethod
        def fromGeoToCartesian(coordinates) : 
          lat,lon=coordinates[:]
          lat,lon=rad(lat),rad(lon)
          x=sin(lat)*cos(lon)
          y=sin(lat)*sin(lon)
          z=cos(lat)
          return [x,y,z]
          
        
        def preparePlotTraj(self,proj='mill',res='i',Dline=5) :
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
            plt.figure()
            map=Basemap(projection=proj,llcrnrlon=self.lons.min(), \
              urcrnrlon=self.lons.max(),llcrnrlat=self.lats.min(),urcrnrlat=self.lats.max(), \
              resolution=res)
            map.drawcoastlines()
#                map.fillcontinents()
            map.drawmapboundary()
            map.drawparallels(self.lats[0::Dline],labels=[1,0,0,0])
            map.drawmeridians(self.lons[0::2*Dline],labels=[0,0,0,1])
            return map
        
        def preparePlotTraj2(self,stateInit,proj='aeqd',res='i',Dline=5,dl=1.5,dh=1) : 
            """
            Prepares the figure to plot a trajectory. Based on basemap.
            
            Parameters
            ----------
            stateInit : array or list:
                The map is based on this position.
            proj : str:
                Name of the projection (default Azimuthal Equidistant)
            res : str:
                Resolution (see Basemap doc)
            Dline : int:
                sampling size for the lats and lons arrays (reduce dimensions)
            dl : float:
                ???
            dh : float:
                ???
                        
            """
            
            plt.figure()
            wdth = (self.lons[-1]-self.lons[0])*dh*math.pi/180*EARTH_RADIUS
            hght = (self.lats[-1]-self.lats[0])*dl*math.pi/180*EARTH_RADIUS
            map = Basemap(width=wdth,height=hght,projection='aeqd',lat_0=stateInit[1],lon_0=stateInit[2],resolution=res)

            map.drawcoastlines()
#                map.fillcontinents()
            map.drawmapboundary()
            map.drawparallels(self.lats[0::Dline],labels=[1,0,0,0])
            map.drawmeridians(self.lons[0::2*Dline],labels=[0,0,0,1])
            return map
        
        def plotTraj(self,states,map,quiv=False,heading=225,scatter=False,color='black'):
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
            x,y=map(posLon,posLat)
            
            if scatter :
                map.scatter(x[0:-1:2],y[0:-1:2],zorder=0,c=color,s=100)
                
            else:
                map.plot(x,y,markersize=4,zorder=0,color=color)
                map.scatter(x[-1],y[-1],zorder=1,color=color)
                
            
            if quiv : 
                points=zip(times,posLat,posLon)
                u=self.uWindAvg(list(points))
                points=zip(times,posLat,posLon)
                v=self.vWindAvg(list(points))
                
                print('u= ' + str(u) + '\n')
                print('v= ' + str(v) + '\n')
                map.quiver(x,y,u,v,zorder=2,width=0.004,color='teal')
                
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
#                    S.set_data(x[instant+t*trajSteps],y[instant+t*trajSteps])
                plt.title('time : ' + str(windAvg.time[instant+t]-windAvg.time[0]) + ' days')       
#                    plt.T.remove()
#                    plt.S.remove()
                return plt
            
            anim = animation.FuncAnimation(fig, update_quiver, frames=range(int(len(states)/3)),fargs=(Q,T,windAvg))
    
            plt.show()

    
            return anim

class Boat : 
      
#        """ 
#        CAREFULL ! if pOfSail>180 we must use pOfSail = 360-pOfSail
#        """
    @staticmethod
    def getDeterDyn(pOfSail,windMag,fitCoeffs) :
        
        if pOfSail > 180 : 
            pOfSail = 360-pOfSail
            
        if windMag > POLAR_MAX_WIND_MAG :
            windMag=POLAR_MAX_WIND_MAG
            
        if pOfSail < POLAR_MIN_POFSAIL :
            speedAtMinPofSail=math.cos(math.pi*POLAR_MIN_POFSAIL/180)* \
                                      np.polynomial.polynomial.polyval2d(POLAR_MIN_POFSAIL,windMag,fitCoeffs)
            return speedAtMinPofSail/(math.cos(math.pi*pOfSail/180))
        
        elif pOfSail>POLAR_MAX_POFSAIL : 
            speedAtMaxPofSail=math.cos(math.pi*POLAR_MAX_POFSAIL/180)* \
                                      np.polynomial.polynomial.polyval2d(POLAR_MAX_POFSAIL,windMag,fitCoeffs)
            return speedAtMaxPofSail/(math.cos(math.pi*pOfSail/180))
        
        else : 
            return np.polynomial.polynomial.polyval2d(pOfSail,windMag,fitCoeffs)
        
    @staticmethod
    def addUncertainty(boatSpeed):
        
        boatSpeedNoisy=rand.gauss(boatSpeed,boatSpeed*BOAT_UNCERTAINTY_COEFF)

        return boatSpeedNoisy


                
                
                
                
                        
                
                
                
                
                
                
                
            
            
            

                
                