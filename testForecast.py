#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:41:29 2017

@author: paul
"""

# basic NOMADS OpenDAP extraction and plotting script
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os
from WeatherClass import Weather
import pickle
import math
import scipy.sparse as sparse
import scipy.io as sio
import BoatClass 
from MDPClass import MDP
# set up the figure
#%%
mydate='20170515'
website='http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs'
#http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs20170506/gfs_0p25_00z
modelcycle='12'
resolution='0p25'
url=website+mydate+'/gfs_'+resolution+'_'+modelcycle+'z'
#os.mkdir('./data/'+mydate)
#pathToSaveObj='./data/'+mydate+'_old.obj'
pathToSaveObj='./data/'+mydate+modelcycle+'.obj'
#latBound=[20,50]
#lonBound=[-80+360, 360]

Weather.download(url,pathToSaveObj)
#
#,latBound=[20,50],lonBound=[-80+360, 360]


#%%