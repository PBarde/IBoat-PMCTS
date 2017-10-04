#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:24:21 2017

@author: paul
"""
import numpy as np
import math
import random as rand
import scipy
from WeatherClass import Weather
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import animation
from math import sin,cos,asin,atan2,acos,pi
from bisect import bisect_left


FIT_VELOCITY=((-0.0310198207067136,-0.0600881995286002,0.0286695485969272, -0.00684406929296715, 0.000379636836557256, -6.77704610076153e-06), \
          (0.0106968590293653, 0.00665508747173206, -4.03686836415123e-05, 2.38962919033178e-05, -6.16724919464073e-07,0), \
          (-0.000370260554113750, -7.68003222256045e-05, -2.44425618051996e-06, -2.16961875441841e-08,0,0),\
          (4.94175336099537e-06, 5.68757177397705e-07, 1.17646387245609e-08,0,0,0),\
          (-2.91900968067076e-08, -1.67287818401469e-09,0,0,0,0),\
          (6.34463723894578e-11,0,0,0,0,0))

FIT_PHI = ((-1.45853032453996 , 4.93261306678728, 2.11068745908768,-0.146421823552765 , 0.00571577488236430,-3.86548180668075e-05 ),\
          ( 0.0450146899230347,-0.265143346808179 , -0.0158879828373159, -6.98702298810359e-05, -2.23691346349847e-05,0 ),\
          ( -6.92983512351375e-05,0.00438845837953799 , 0.000149659040461729, 4.59832959915573e-06, 0,0 ),\
          ( -8.07833836390101e-06, -3.22527331822989e-05,-7.32725492793235e-07 , 0, 0, 0),\
          ( 9.32065359630392e-08, 8.58978734563361e-08, 0, 0, 0, 0),\
          ( -2.99522066048220e-10, 0, 0, 0, 0, 0))

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
DESTINATION_RADIUS=0.025*pi/180*EARTH_RADIUS
                        
class Simulator : 
    
            def __init__(self,times,lats,lons,WeatherAvg,WeatherEns,stateInit) :
                
                self.times=times
                self.lats=lats
                self.lons=lons
                self.state=list(stateInit)
        
                WeatherAvg.Interpolators()
                self.uWindAvg=WeatherAvg.uInterpolator
                self.vWindAvg=WeatherAvg.vInterpolator
                
                WeatherEns.Interpolators()
                self.uWindSpr=WeatherEns.uInterpolator
                self.vWindSpr=WeatherEns.vInterpolator
             
            def reset(self,stateInit):
                
                self.state=list(stateInit)
                
            
            def getDistAndBearing(self,position,destination):
                latDest,lonDest=[destination[0],destination[1]]
                latDest=latDest*math.pi/180
                lonDest=lonDest*math.pi/180
                
                latPos=position[0]*math.pi/180
                lonPos=position[1]*math.pi/180
                a=(sin((latDest-latPos)/2))**2+cos(latPos)*cos(latDest)*(sin((lonDest-lonPos)/2))**2
                
                distance=2*EARTH_RADIUS*atan2(a**0.5,(1-a)**0.5)
                
                x=math.cos(latDest)*math.sin(lonDest-lonPos)
                y=math.cos(latPos)*math.sin(latDest)-math.sin(latPos)*math.cos(latDest)\
                          *math.cos(lonDest-lonPos)
                          
                bearing=(math.atan2(x,y)*180/math.pi+360)%360
                
#                print(str(bearing))
                return distance,bearing
    

            def getDestination(self,distance,bearing,departure):
                latDep,lonDep=[departure[0],departure[1]]
                
                latDep=latDep*math.pi/180
                lonDep=lonDep*math.pi/180
                
                bearing=bearing*math.pi/180
                
                latDest=asin(sin(latDep)*cos(distance/EARTH_RADIUS) + \
                                 cos(latDep)*sin(distance/EARTH_RADIUS)*cos(bearing))
                
                lonDest=lonDep + atan2(sin(bearing)*sin(distance/EARTH_RADIUS) \
                                            *cos(latDep),cos(distance/EARTH_RADIUS)\
                                                     -sin(latDep)*sin(latDest))
                
                latDest=(latDest*180/math.pi)
                lonDest=(lonDest*180/math.pi)
#                print(str(bearing))
                                  
                return [latDest,lonDest]
            
            def getWind(self) :
                uAvg=self.uWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
                vAvg=self.vWindAvg([self.times[self.state[0]],self.state[1],self.state[2]])
                
                uSpr=self.uWindSpr([self.times[self.state[0]],self.state[1],self.state[2]])
                vSpr=self.vWindSpr([self.times[self.state[0]],self.state[1],self.state[2]])
                
                uWind=rand.gauss(uAvg,uSpr)
                vWind=rand.gauss(vAvg,vSpr)
                
                return uWind,vWind
            
            """ CAREFULLL : ONLY VALID FOR SMALL TIME STEPS (1HOUR) 
            else Dlat and Dlon might be wrong and we need to use the getPolarVel and getDestination method""" 
            def doStep(self,action):
                
#                action=Simulator.takeClosest(ACTIONS_DP,action)
#                print(action)
                #we get the wind at current state
                uWind,vWind=self.getWind()
                windMag,windAng=Weather.returnPolarVel(uWind,vWind)
                
                #We get speed from wind on sail
                pOfSail=abs((windAng+180)%360-action)
                boatSpeed=Boat.getDeterDyn(pOfSail,windMag,FIT_VELOCITY)
                uBoat,vBoat = Boat.addUncertainty(action,boatSpeed)
                
                #We get speed from current
                uBoat,vBoat=Current.addSpeed(uBoat,vBoat,uWind,vWind)
#                print('(uBoat,vBoat)=(' + str(uBoat)+', '+ str(vBoat)+')\n')
                #We integrate it
                Dt=(self.times[self.state[0]+1]-self.times[self.state[0]])*DAY_TO_SEC
#                print('Dt=' + str(Dt) +'\n')
                """ This is correct for big timeSteps"""
                vMag,vAng=Weather.returnPolarVel(uBoat,vBoat)
                DL=vMag*Dt
                newLat,newLon=self.getDestination(DL,vAng,[self.state[1],self.state[2]])
                
#                DLon=uBoat*Dt/(EARTH_RADIUS*math.cos(self.state[1]*math.pi/180))*180/math.pi
#                DLat=vBoat*Dt/EARTH_RADIUS*180/math.pi
#                
#                newLat=self.state[1]+DLat
#                newLon=self.state[2]+DLon
                                 
                self.state[0],self.state[1],self.state[2]=self.state[0]+1,newLat,newLon
#                print('state=' + str(self.state) +'\n')
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
            
            
            
            def praparePlotTraj(self,proj='mill',res='i',Dline=5) : 
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
            
            def praparePlotTraj2(self,stateInit,proj='aeqd',res='i',Dline=5,dl=1.5,dh=1) : 
                plt.figure()
                map = Basemap(width=(self.lons[-1]-self.lons[0])*dh*math.pi/180*EARTH_RADIUS,height=(self.lats[-1]-self.lats[0])*dl*math.pi/180*EARTH_RADIUS,\
                              projection='aeqd',lat_0=stateInit[1],lon_0=stateInit[2],resolution=res)

                map.drawcoastlines()
#                map.fillcontinents()
                map.drawmapboundary()
                map.drawparallels(self.lats[0::Dline],labels=[1,0,0,0])
                map.drawmeridians(self.lons[0::2*Dline],labels=[0,0,0,1])
                return map
            
            def plotTraj(self,states,map,quiv=False,line=False,heading=225,scatter=False,color='black'):
                states=np.array(states)
                posLat=states[:,1]
                posLon=states[:,2]
                times=self.times[states[:,0].astype(int)]
                x,y=map(posLon,posLat)
                
                if scatter :
                    map.scatter(x[0:-1:2],y[0:-1:2],zorder=0,color='goldenrod',s=100)
                    
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
                    
                if line:
                    latI,lonI=states[0,1],states[0,2]
                    length=((x.max()-x.min())**2+(y.max()-y.min())**2)**0.5
                    latF,lonF=self.getDestination(length,heading,[latI,lonI])
                    xline,yline=map([lonI,lonF],[latI,latF])
                    map.plot(xline,yline,color='black')
                    
                if quiv : 
                    return u,v
                
            def animateTraj(self,windAvg, states, trajSteps=3, proj='mill', res='i', instant=0, Dline=5, density=1, interval=1): 
                """
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
        def addUncertainty(heading,boatSpeed):
            uBoat=boatSpeed*math.sin(heading*math.pi/180)
            vBoat=boatSpeed*math.cos(heading*math.pi/180)
            uBoat=rand.gauss(uBoat,boatSpeed*BOAT_UNCERTAINTY_COEFF)
            vBoat=rand.gauss(vBoat,boatSpeed*BOAT_UNCERTAINTY_COEFF)
            
            return uBoat,vBoat
            
            
             
                
            
CURRENT_SCALING_FROM_WIND = 0.03
#CURRENT_SCALING_FROM_WIND = 0.

class Current :
    
        @staticmethod
        def addSpeed(uBoat,vBoat,uWind,vWind) :
            uBoat=uBoat+uWind*CURRENT_SCALING_FROM_WIND
            vBoat=vBoat+vWind*CURRENT_SCALING_FROM_WIND
            return uBoat,vBoat      


                
                
                
                
                        
                
                
                
                
                
                
                
            
            
            

                
                