#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:36:10 2017

@author: paul
"""


import sys
import MyTree as mt
sys.path.append("../model")
import pickle
import sys
from WeatherTLKT import Weather
import numpy as np
from SimulatorTLKT import Simulator
import matplotlib.pyplot as plt
import matplotlib
from MyTree import Tree
matplotlib.rcParams.update({'font.size': 16})



HOURS_TO_DAY = 1 / 24
# %% We load the forecast files
mydate = '20170519'
modelcycle = '00'
pathToSaveObj = './data/' + mydate + '_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)

# %% We shift the times so that all times are in the correct bounds for interpolations

Tini = Wavg.time[0]

Wavg.time = Wavg.time - Tini

# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)
Tf = 24 * 5
times = np.arange(0, Tf * HOURS_TO_DAY, 1 * HOURS_TO_DAY)
lats = np.arange(Wavg.lat[0],Wavg.lat[-1], 1.05)
lons = np.arange(Wavg.lon[0], Wavg.lon[-1], 1.05)

stateInit = [0, 47.5, -3.5 + 360]

Sim = Simulator(times, lats, lons, Wavg, stateInit)


# %% We set up the parameters of the simulation
# times=np.arange(0,min([Wavg.time[-1],Wspr.time[-1]]),1*HOURS_TO_DAY)
# Tf=len(times)

heading = 235
dests = []
dists = []

ntra = 50


for traj in range(ntra):
    Sim.reset(stateInit)
    tra = []
    for t in Sim.times[0:-1]:
        tra.append(list(Sim.doStep(heading)))
    dests.append(list(Sim.state))
    d, dump = Sim.getDistAndBearing(Sim.state[1:],[stateInit[1], stateInit[2]])
    dists.append(d)

dmean=np.mean(dists)
dests=np.array(dests)
latDest=np.mean(dests[:,1])
lonDest=np.mean(dests[:,2])
destination = [latDest,lonDest]
Times = []
dests2=[]
for traj in range(ntra):
    Sim.reset(stateInit)
    tra2 = []
    d = dmean
    dist, action = Sim.getDistAndBearing(Sim.state[1:],destination)
    tra2.append(list(Sim.doStep(action)))
    atDest,frac =Tree.isStateAtDest(destination,Sim.prevState,Sim.state)
    dist, action = Sim.getDistAndBearing(Sim.state[1:],destination)
    
    while (not atDest) \
                and (not Tree.isStateTerminal(Sim,Sim.state)):
            tra2.append(list(Sim.doStep(action)))
            dist, action = Sim.getDistAndBearing(Sim.state[1:],destination)
            atDest,frac =Tree.isStateAtDest(destination,Sim.prevState,Sim.state)
            
    if atDest:
      finalTime = Sim.times[Sim.state[0]]-(1-frac)
      Times.append(finalTime)
      dests2.append(list(Sim.state))
      
      
dests2=np.array(dests2)
Tmean = np.mean(Times)
print('Number of Boat that arrived : ' + str(len(Times)))
#%%
m = Sim.prepareBaseMap(centerOfMap=stateInit[1:],proj='aeqd')
# Azimuthal Equidistant Projection
#The shortest route from the center of the map to any other point is a straight line in the azimuthal equidistant
#projection. So, for the specified point, all points that lie on a circle around this point are equidistant
#on the surface of the earth on this projection.
rgba_colors = np.zeros((len(dests2),4))
rgba_colors[:, 3] = np.exp(min(Times)-np.array(Times))


x1, y1 = m(stateInit[2], stateInit[1])
x2, y2 = m(destination[1], destination[0])
m.scatter([x1, x2], [y1, y2])
#Sim.plotTraj(dests, m,scatter=True)
#Sim.plotTraj(tra, m)
#Sim.plotTraj(tra2, m, color='red')
Sim.plotTraj(dests2, m, scatter=True,color=rgba_colors)


#%%
n=len(Times)

m = Sim.prepareBaseMap(centerOfMap=destination,proj='aeqd',dl=.005,dh=.005)
# Azimuthal Equidistant Projection
#The shortest route from the center of the map to any other point is a straight line in the azimuthal equidistant
#projection. So, for the specified point, all points that lie on a circle around this point are equidistant
#on the surface of the earth on this projection.
rgba_colors = np.zeros((len(dests2[:n]),4))
rgba_colors[:, 3] = np.exp(min(Times[:n])-np.array(Times[:n]))


x1, y1 = m(stateInit[2], stateInit[1])
x2, y2 = m(destination[1], destination[0])
m.scatter([x1, x2], [y1, y2])


xi,yi=m(dests2[:n,2],dests2[:n,1])
m.scatter(xi,yi,c=rgba_colors)
for i in range(len(Times[:n])):
    plt.text(xi[i],yi[i],'t=' + "%.2f"%Times[i])
#    plt.text(xi[i],yi[i], "%.2f"%rgba_colors[i, 3])
#xline,yline=m([tra[0][2],tra[-1][2]],[tra[0][1],tra[-1][1]])
#m.plot(xline,yline,color='blue')
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
