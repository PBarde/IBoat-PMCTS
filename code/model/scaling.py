#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:48:40 2017

@author: paul
"""
import math
from decimal import Decimal

V_MAX_BOAT = 3  #""" in m/s """
EARTH_RADIUS = 6371e3 #""" in m """
#GRID_RES=0.25 * math.pi/180  #""" in deg """
GRID_RES=1 * math.pi/180  #""" in deg """
HORIZON = 6 #""" in days """
#MDP_TIME_STEP = 3*60*60 #in s
MDP_TIME_STEP = 6*60*60 #in s

MtoNM=1/1852
MpStoKnots=1.943844

timeHorizon=HORIZON*24*60*60

distance=V_MAX_BOAT*timeHorizon
distTimeStep=V_MAX_BOAT*MDP_TIME_STEP

angularDist=distance/EARTH_RADIUS
angularDistTimeStep=distTimeStep/EARTH_RADIUS

gridSpan = math.ceil(angularDist/GRID_RES)
gridSpanStep = angularDistTimeStep/GRID_RES
timeSpan = math.ceil(timeHorizon/MDP_TIME_STEP)
nStates = timeSpan*gridSpan**2

print('Grid resolution : ' + str(GRID_RES*180/math.pi) + 'deg / ' + str(GRID_RES*EARTH_RADIUS*MtoNM) + 'nautic miles')
print(' \n Number of grid cells covered at Vmax in ' + str(HORIZON) +' days : ' + str(gridSpan) + ' cells\n' )
print(' Number of grid cells covered at Vmax in time step : ' + str(gridSpanStep) + ' cells\n' )
print(' Number of executed time steps in ' + str(HORIZON) +' days : ' + str(timeSpan) + ' steps\n' )
print(' Displacement in ' + str(HORIZON) +' days : ' + str(angularDist*180/math.pi) + ' deg / ' + str(angularDist*EARTH_RADIUS*MtoNM) + 'Nm\n' )

print(' Total number of states : ' + str('%.2E' % Decimal(nStates)) + ' states')
