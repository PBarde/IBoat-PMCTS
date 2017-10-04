#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:09:26 2017

@author: paul
"""
import numpy as np

class MDP :
        def __init__(self) :
            self.States=None
            self.Actions=None
            self.Transitions=None
            self.Rewards=None
            self.Discount=None
            self.Ntime=None
            self.Nlat=None
            self.Nlon=None
            self.h=None
           
        def setMDPcharacs(self, WeatherObj, Actions, times, lats, lons, destination, stateCost=-1, finalReward=100) :
            
            #setting up the grid for finding closest point
            lat_lon_arrays=np.dstack([lats.ravel(),lons.ravel()])[0]
            
            
            self.Ntime=np.size(times)
            self.Nlat=np.size(lats)
            self.Nlon=np.size(lons)
            WeatherObj.Interpolators()
            
            # We set up the actions
            nActions=np.size(Actions)
            self.Actions=Actions
            
            # We set up the states and the rewards
            self.States=[]
            self.Rewards=[]
            
            for t in range(self.Ntime):
                for i in range(self.Nlon):
                    for j in range(self.Nlat):
                        self.States.append((WeatherObj.time[t],WeatherObj.lon[i],WeatherObj.lat[j]))
                        self.Rewards.append(stateCost)
            
            