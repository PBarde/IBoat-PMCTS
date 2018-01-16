#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:53:03 2018

@author: paul
"""
import forest as ft
from simulatorTLKT import Simulator, HOURS_TO_DAY
import numpy as np
from worker import Tree
mydate = '20180108'

#ft.Forest.download_scenarios(mydate,latBound = [50-28, 50],lonBound = [-40 + 360, 360])
#
Weathers = ft.Forest.load_scenarios(mydate, latBound = [40, 50],lonBound = [360-15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 4  # <=20
SIM_TIME_STEP = 6 # in hours
STATE_INIT = [0, 47.5, -3.5 + 360]
N_DAYS_SIM = 8 # time horizon in days

sims = ft.Forest.create_simulators(Weathers, numberofsim = NUMBER_OF_SIM, simtimestep = SIM_TIME_STEP,
                                   stateinit = STATE_INIT, ndaysim = N_DAYS_SIM)


# initialize the simulators to get common destination and individual time min

missionheading = 235
ntra = 50

destination, timemin, meantrajs, meantrajs_dest = ft.Forest.initialize_simulators(sims,ntra,STATE_INIT,missionheading, plot = True)

print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")

#todo plotter tout les mean trajs pour chaque scénarios avec le vent observé, le point de départ et 
#la destination  
#%%
#weather=Weathers[0]
#weather.getPolarVel()
#mapp=weather.plotQuiver()
#
#x,y=mapp(destination[1],destination[0])
#plt.scatter(x,y)


#%%
#def box_pir(l,destination):
#    lengths = [i for i in map(len, l)]
#    shape = (len(l), max(lengths))
#    a = np.empty(shape,dtype=object)
#    a.fill(np.array(destination))
#    for i, r in enumerate(l):
#        a[i, :lengths[i]] = np.array(r)
#    return a
#
#t1 = [[1,2],[1,2],[1,2]]
#t2 = [[2,4],[2,4]]
#
#t=[t1,t2]
#
#tt = box_pir(t,[1,1])
