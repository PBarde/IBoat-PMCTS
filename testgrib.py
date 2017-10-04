#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:21:44 2017

@author: paul
"""
from WeatherClass import Weather

mydate = '20170427'
website = 'http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs'
modelcycle = '00'
resolution = '0p25'
url = website + mydate + '/gfs_' + resolution + '_' + modelcycle + 'z'
# %%
Weather.download(url=url, folder='./data', new=True, newFolder='/' + mydate)
# %%
mydate = '20170427'
website = 'http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs'
modelcycle = '00'
resolution = '0p25'
url = website + mydate + '/gfs_' + resolution + '_' + modelcycle + 'z'
Weather.download(url=url, folder='./data')
# %%
forecast1 = Weather('./gfs_' + resolution + '_' + modelcycle + 'z')
