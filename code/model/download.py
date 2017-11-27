#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:33:38 2017

@author: paul
"""

from WeatherTLKT import Weather

typ='ens'

for ss in range(1,9):
  if typ=='solo':
    mydate='20171127'
    website='http://nomads.ncep.noaa.gov:9090/dods'
    model='gfs'
    resolution='0p25'
    url=website+'/'+model+'_'+resolution+'/'+model+mydate+'/'+model+'_'+resolution+'_00z'
    pathToSaveObj='../data/'+ model+mydate+'_'+resolution
  
  else : 
    mydate='20171127'
    website='http://nomads.ncep.noaa.gov:9090/dods'
    model='gens'
    resolution='0p25'
    num_scenario='0'+str(ss)
    url=website+'/'+model+'/'+model+mydate+'/'+'gep'+num_scenario+'_00z'
    pathToSaveObj='../data/'+ model+mydate+'_'+num_scenario
  
  latBound=[43,50]
  lonBound=[-10+360, 360]
  
  
  Weather.download(url,pathToSaveObj,latBound=latBound,lonBound=lonBound,timeSteps=[0,85],ens=True)

