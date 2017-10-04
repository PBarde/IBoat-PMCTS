#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:55:41 2017

@author: paul
"""

import mdptoolbox
import mdptoolbox.example
import numpy as np

lats=np.arange(0,-10,-0.5)
lons=np.arange(0,20,0.5)
times=np.arange(0,1,0.1)

LONS,LATS,TIMES=np.meshgrid(lons,lats, times, indexing='xy')
STATES=np.dstack([LONS.ravel(),LATS.ravel(),TIMES.ravel()])[0]


#%%


