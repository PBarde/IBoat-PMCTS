#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:33:38 2017

@author: paul
"""

from WeatherClass import Weather


mydate='20171015'
website='http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs20171015/gfs_0p25_00z'
#modelcycle='00'
#resolution='0p25'
url=website
#+mydate+'/gfs_'+resolution+'_'+modelcycle+'z'
#url='http://nomads.ncep.noaa.gov:9090/dods/wave/nww3/nww320170519/nww320170519_00z'
#os.mkdir('./data/'+mydate)
latBound=[43,50]
lonBound=[-10+360, 360]
pathToSaveObj='./data/'+ 'test'
#mydate+'_'+modelcycle+'.obj'

Weather.download(url,pathToSaveObj,latBound=latBound,lonBound=lonBound,timeSteps=[0,81])


website='http://nomads.ncep.noaa.gov:9090/dods/gens_bc/gens20170926/gespr_00z'
url=website
#+mydate+'/gespr_'+modelcycle+'z'
pathToSaveObj='./data/'+ 'test2'
#mydate+'_'+modelcycle+'ens.obj'
Weather.download(url,pathToSaveObj,latBound=latBound,lonBound=lonBound)
