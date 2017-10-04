#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:36:10 2017

@author: paul
"""

import MyTree as mt
import numpy as np
from WeatherClass import Weather
import numpy as np
from SimulatorClass import Simulator
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import SimulatorClass as SimC
import pickle
import sys

HOURS_TO_DAY = 1 / 24
# %% We load the forecast files
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = './data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)


pathToSaveObj = './data/' + mydate + '_' + modelcycle + 'ens.obj'
Wspr = Weather.load(pathToSaveObj)
# we crop the Nan values
Wspr = Wspr.crop(timeSteps=[1, 64])
# %% We shift the times so that all times are in the correct bounds for interpolations

Tini = max([Wavg.time[0], Wspr.time[0]])
Wspr.time = Wspr.time - Tini
Wavg.time = Wavg.time - Tini
Tf = 24 * 2
#SimC.DESTINATION_RADIUS=SimC.DESTINATION_RADIUS*3
times = np.arange(0, Tf * HOURS_TO_DAY, 1 * HOURS_TO_DAY)
lats = np.arange(max([Wavg.lat[0], Wspr.lat[0]]), min([Wavg.lat[-1], Wspr.lat[-1]]), 0.05)
lons = np.arange(max([Wavg.lon[0], Wspr.lon[0]]), max([Wavg.lon[-1], Wspr.lon[-1]]), 0.05)

stateInit = [0, 47.5, -3.5 + 360]


# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)

Sim = Simulator(times, lats, lons,  Wavg, Wspr, stateInit)
heading = 235
dests = []
dists = []
dists2 = []
ntra = 50

for traj in range(ntra):
    Sim.reset(stateInit)
    tra = []
    for t in Sim.times[0:-1]:
        tra.append(list(Sim.doStep(heading)))
    dests.append(list(Sim.state))
    d, dump = Sim.getDistAndBearing(Sim.state[1:],[stateInit[1], stateInit[2]])
    dists.append(d)

# dmean=np.mean(dists)
dmean = min(dists)*3/4
destination = Sim.getDestination(dmean, heading, [stateInit[1], stateInit[2]])
Times = []

for traj in range(ntra):
    Sim.reset(stateInit)
    tra2 = []
    d = dmean
    t=0
    while t < len(Sim.times)-1 and d > SimC.DESTINATION_RADIUS:
        d, dump = Sim.getDistAndBearing(Sim.state[1:],destination)
        tra2.append(list(Sim.doStep(dump)))
        t=t+1
        
    if d < SimC.DESTINATION_RADIUS :
        Times.append(Sim.times[t])

Tmin = min(Times)
print('Number of Boat that arrived : ' + str(len(Times)))
#%%
#plt.figure()
m = Sim.praparePlotTraj2(stateInit,dl=0.35,dh=0.25)
x1, y1 = m(stateInit[2], stateInit[1])
x2, y2 = m(destination[1], destination[0])
m.scatter([x1, x2], [y1, y2])
Sim.plotTraj(dests, m,scatter=True)
Sim.plotTraj(tra, m,quiv=True,line=True,heading=235)
Sim.plotTraj(tra2, m)
m.fillcontinents()
# %%

Tmin=1.875
destination=[47.09030923512784, 355.6472848111828]
mt.UCT_COEFF=1/2**0.5
stateInit = [0, 47.5, -3.5 + 360]
Sim = Simulator(times, lats, lons, Wavg, Wspr, stateInit)
tree = mt.Tree(destination=destination, simulator=Sim, budget=15000, TimeMin=Tmin)
tree.UCTSearch(stateInit)
# tree.plotTree()
##%%
#
#    
# UTCs=[]
# for child in tree.rootNode.children :
#    UTCs.append(tree.getUCT(child, UCT_COEFF))
# print(str(UTCs)+'\n')
# max_UTC = max(UTCs)
# max_index = UTCs.index(max_UTC)
#
# print( str(tree.rootNode.children[max_index])+'\n')
# print(max_index)
# %%
sys.setrecursionlimit(100000)
coeff=str(1/mt.UCT_COEFF)[:3]
name=coeff+'UCT_theTree_heading_'+str(heading)+'_b'+str(tree.budget)+'_'+str(Tf/24)+'days_ContinuousDP2'
#%%
filehandler = open(name+'.pickle', 'wb')
pickle.dump(tree, filehandler)
filehandler.close()
#tree.plotTree()
#plt.title(name)
#%%
filehandler=open(name+'.pickle','rb')
loaded=pickle.load(filehandler)
filehandler.close()
listOfFigs=loaded.plotBD(3)
#loaded.plotTree()

#%%
k=1
for fig in listOfFigs:
    ax=fig.gca()
    ax.set_xlim([-20,4])
    ax.set_ylim([-7,9])
    ax.grid()
    fig.savefig('./ResultsBD/'+str(k)+'BD_'+name+'.pdf', bbox_inches='tight')
    k=k+1
    fig.show()
#%%
filehandler = open('BD_'+name+'.pickle', 'wb')
pickle.dump(listOfFigs, filehandler)
filehandler.close()
#fig=listOfFigs[2]
#ax=fig.gca()
#ax.set_xlim([-20,4])
#ax.set_ylim([-7,9])
##plt.figure().canvas.draw()
#fig.show()

#SimC.DESTINATION_RADIUS=SimC.DESTINATION_RADIUS/3




    
#
#
##
##%%
#Sim=Simulator(times,lats,lons,Wavg,Wspr,stateInit)
#optimalActions=[]
##loaded=tree
#node=loaded.rootNode
#
#while node.children :
#   bestChild=loaded.bestChild(node,0)
#   optimalActions.append(bestChild.origin)
#   node=bestChild
#    
#nTraj=200
#propAct=1/2
#finalTimeOpt=[]
#for traj in range(nTraj):
#   Sim.reset(stateInit)
#   for action in optimalActions[0:int(len(optimalActions)*propAct)] : 
#       Sim.doStep(action)
#       if loaded.isStateAtDest(Sim.state) or loaded.isStateTerminal(Sim.state) : 
#           break
#        
#   while (not loaded.isStateAtDest(Sim.state)) \
#                and (not loaded.isStateTerminal(Sim.state)) :
#               dist,action=Sim.getDistAndBearing(Sim.state[1:],loaded.destination)
#               Sim.doStep(action)
#   finalTimeOpt.append(int(Sim.state[0]))
#    
#print('Mean arrival date optimal path : ' + str(np.mean(finalTimeOpt)) + ' hours')
#finalTimeDir=[]
#
#for traj in range(nTraj):
#   Sim.reset(stateInit)
#        
#   while (not loaded.isStateAtDest(Sim.state)) \
#                and (not loaded.isStateTerminal(Sim.state)) :
#               dist,action=Sim.getDistAndBearing(Sim.state[1:],loaded.destination)
##                print(str(action))
#               Sim.doStep(action)
#   finalTimeDir.append(int(Sim.state[0]))
#    
#print('Mean arrival date direct path : ' + str(np.mean(finalTimeDir)) + ' hours')
##
#""" Trouver pourquoi ça marche mieux avec le UCT avec N ite total"""

#Mean arrival date optimal path : 41.73 days
#Mean arrival date direct path : 42.265 days


#%%
#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib
#import math
#matplotlib.rcParams.update({'font.size': 26})
#Wspr.getPolarVel()
#mu = 15
#variances = Wspr.wMag[0:16:4,0,0]**2
#times=Wspr.time[0:16:4]-Wspr.time[0]
#legend=[]
#for t in times :
#    legend.append(str(int(t)) + ' days')
#
#fg=plt.figure()
#
#for variance in variances :
#    sigma = math.sqrt(variance)
#    x = np.linspace(mu-5*variance,mu+5*variance, 1000)
#    plt.plot(x,mlab.normpdf(x, mu, sigma),linewidth=4)
#
#plt.grid()
#plt.legend(legend)
#plt.xlim([7.5,22.5])
#plt.xlabel('Wind speed [m/s]')
#plt.ylabel('PDF [s/m]')
#plt.show()
#fg.savefig('../../../Article/Figures/standard_deviation.pdf', bbox_inches='tight')
 
##%%
##rootHash=mt.Node.getHash(stateInit)
# fig=plt.figure()
# map=tree.Simulator.praparePlotTraj()
##mt.Tree.plotChildren(tree.nodes[rootHash],map)
# nodeList=list(tree.nodes.values())
# for node in nodeList :
##    print(node.N)
#    x,y=map(node.state[2],node.state[1])
#    map.scatter(x,y,zorder=0)
#
# pickle.dump(fig, open('FigureObject_veryV5.fig.pickle', 'wb'))
##%%
# plt.figure()
# map2=tree.Simulator.praparePlotTraj()
##mt.Tree.plotChildren(tree.nodes[rootHash],map)
#
# for state in tree.listOfFinalStates :
#    x,y=map2(state[2],state[1])
#    map2.scatter(x,y,zorder=0)
#    
##%%
# plt.figure()
# map3=tree.Simulator.praparePlotTraj()
# mt.Tree.plotChildren(nodeList[0],map3)
##%%
#
#
# figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
#
# figx.show() # Show the figure, edit it, etc.!
