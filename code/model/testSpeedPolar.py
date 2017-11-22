#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:03:27 2017

@author: paul
"""

 
from SimulatorTLKT import Boat
from SimulatorTLKT import FIT_VELOCITY
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi

matplotlib.rcParams.update({'font.size': 22})
pOfS=np.arange(0,360,0.5)
wMags=np.arange(0,25,2)
polars=[]
legends=[]
fig=plt.figure()
for mag in wMags: 
    pol=[]
    legends.append('Wind mag = '+str(mag) + ' m/s')
    for p in pOfS :
        pol.append(Boat.getDeterDyn(p,mag,FIT_VELOCITY))
    polars.append(list(pol))
    ax=plt.polar(pOfS*pi/180,pol,label=str(mag) + ' m/s')
#plt.legend(legends)
plt.legend(bbox_to_anchor=(1.1,1), loc=2, borderaxespad=0.)
#plt.xlabel('Polar plot of Boat velocity [m/s] wrt. point of sail [deg]',fontsize=22)
#ax.xaxis.set_label_position('top')

fig.savefig('../../../Article/Figures/polar_modified2.pdf', bbox_inches='tight')
    
