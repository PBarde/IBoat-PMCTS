#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:53:03 2018

@author: paul
"""
import forest as ft
from utils import HOURS_TO_DAY
from simulatorTLKT import Simulator
import numpy as np
mydate = '20180108'

# ft.Forest.download_scenarios(mydate)

Weathers = ft.Forest.load_scenarios(mydate)

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 4  # <=20
SIM_TIME_STEP = 6 # in hours
STATE_INIT = [0, 47.5, -3.5 + 360]
N_DAYS_SIM = 8 # time horizon in days
DELTA_LAT_LON = 0.05 # in degrees

sims = []
for jj in range(NUMBER_OF_SIM):
    # We shift the times so that all times are in the correct bounds for interpolations
    Weathers[jj].time = Weathers[jj].time - Weathers[jj].time[0]

    # We set up the parameters of the simulation
    timemax = SIM_TIME_STEP * len(Weathers[jj].time)
    times = np.arange(0, N_DAYS_SIM, SIM_TIME_STEP * HOURS_TO_DAY)
    lats = np.arange(Weathers[jj].lat[0], Weathers[jj].lat[-1], DELTA_LAT_LON)
    lons = np.arange(Weathers[jj].lon[0], Weathers[jj].lon[-1], DELTA_LAT_LON)
    sims.append(Simulator(times, lats, lons,Weathers[jj],STATE_INIT))





