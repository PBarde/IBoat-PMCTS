#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:34:50 2017

@author: paul
"""
import sys
sys.path.append("../model")
import MyTree as mt
import numpy as np
from WeatherClass import Weather
import numpy as np
from SimulatorClass import Simulator
import SimulatorClass as SimC
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
import sys
from math import sqrt
#%%
#test des familles
heading=235
budget=15000
nDays=2.0
coeff=str(sqrt(2))[:3]
name=coeff+'UCT_theTree_heading_'+str(heading)+'_b'+str(budget)+'_'+str(nDays)+'days'

#%%
filehandler=open(name+'Results10Actions.pickle','rb')
[listOfTrajOptim,listOfTrajDirect,finalTimes,ulistOptim,vlistOptim,\
                                ulistDirect,vlistDirect]=pickle.load(filehandler)
filehandler.close()
#%%
filehandler=open(name+'.pickle','rb')
loaded=pickle.load(filehandler)
filehandler.close()
print('tree depth : '+str(loaded.depth))

#%%
trajsOptim=np.array(listOfTrajOptim)
meantrajOptim=np.mean(trajsOptim,axis=0)

times=[]
for traj in listOfTrajDirect : 
    times.append(traj[-1][0])

lT=int(np.mean(times)-10)


trajsDirectCut=[]
for traj in listOfTrajDirect : 
    if len(traj)>=lT: 
        trajsDirectCut.append(list(traj[:lT]))

trajsDirect=np.array(trajsDirectCut)    
meantrajDirect=np.mean(trajsDirect,axis=0)
#%%
plt.figure()
m = loaded.Simulator.praparePlotTraj2(loaded.rootNode.state,dl=0.3,dh=0.2)

x1, y1 = m(loaded.rootNode.state[2], loaded.rootNode.state[1])
m.scatter(x1,y1,color='blue')
m.fillcontinents()
loaded.Simulator.plotTraj(meantrajOptim, m,color='red',quiv=True)
loaded.Simulator.plotTraj(meantrajDirect, m,quiv=True)

x2, y2 = m(meantrajDirect[-1][2], meantrajDirect[-1][1])
m.scatter(x2,y2,color='grey')


#%%
m = loaded.Simulator.praparePlotTraj2(loaded.rootNode.state,dl=0.3,dh=0.2)
x2, y2 = m(loaded.destination[1], loaded.destination[0])
m.scatter(x2,y2,color='red')
x1, y1 = m(loaded.rootNode.state[2], loaded.rootNode.state[1])
m.scatter(x1,y1,color='blue')
m.fillcontinents()

for trajOptim,trajDirect in zip(listOfTrajOptim[:100],listOfTrajDirect[:100]) : 
    loaded.Simulator.plotTraj(trajOptim, m,color='red')
    loaded.Simulator.plotTraj(trajDirect, m)
#%%
heading=235
budget=15000
nDays=2.0
coeff=str(sqrt(2))[:3]
name=coeff+'UCT_theTree_heading_'+str(heading)+'_b'+str(budget)+'_'+str(nDays)+'days'

#%%

filehandler=open('Results10Actions.pickle','rb')
[listOfTrajOptim,listOfTrajDirect,finalTimes,ulistOptim,vlistOptim,\
                                ulistDirect,vlistDirect]=pickle.load(filehandler)
filehandler.close()                      
#%%
for trajDirect in listOfTrajDirect[:100] : 
    loaded.Simulator.plotTraj(trajDirect, m,color='green')

#%%
times=[]
for traj in listOfTrajDirect : 
    times.append(traj[-1][0])

lT=int(np.mean(times))


trajsDirectCut=[]
for traj in listOfTrajDirect : 
    if len(traj)>=lT: 
        trajsDirectCut.append(list(traj[:lT]))

trajsDirect=np.array(trajsDirectCut)    
meantrajDirect=np.mean(trajsDirect,axis=0)
loaded.Simulator.plotTraj(meantrajDirect, m, color='green',quiv=True)
