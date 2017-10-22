#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:49:31 2017

@author: paul
"""

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

heading=235
budget=15000
nDays=2.0
coeff=str(sqrt(2))[:3]
name=coeff+'UCT_theTree_heading_'+str(heading)+'_b'+str(budget)+'_'+str(nDays)+'days_ContinuousDP'
#%%
filehandler=open(name+'.pickle','rb')
loaded=pickle.load(filehandler)
filehandler.close()
print('tree depth : '+str(loaded.depth))
#%%
listOfActions=[]
node=loaded.rootNode

while node.children:
    bestChild=loaded.bestChild(node,0)
    listOfActions.append(bestChild.origin)
    node=bestChild
#%%
listOfTrajOptim=[]
listOfTrajDirect=[]
nTra=2390
finalTimes=[]
ulistOptim=[]
vlistOptim=[]
m = loaded.Simulator.praparePlotTraj2(loaded.rootNode.state,dl=0.3,dh=0.2)

for tra in range(nTra) : 
    loaded.Simulator.reset(loaded.rootNode.state)
    listOfStateOptim=[]
    
    for action in listOfActions[:10]:
        state=list(loaded.Simulator.doStep(action))
        listOfStateOptim.append(state)
        if loaded.isStateAtDest(state) or loaded.isStateTerminal(state) : 
            break

    u,v=loaded.Simulator.plotTraj(listOfStateOptim, m,quiv=True)
    ulistOptim.append(u)
    vlistOptim.append(v)
    
    ulistDirect=[]
    vlistDirect=[]
    listOfStateDirect=[]
#    state=list(loaded.Simulator.state)
    while not loaded.isStateAtDest(state) and not loaded.isStateTerminal(state) :
        d,action = loaded.Simulator.getDistAndBearing(state[1:],loaded.destination)
        state=list(loaded.Simulator.doStep(action))
        listOfStateDirect.append(state)
        
    u,v=loaded.Simulator.plotTraj(listOfStateDirect, m,quiv=True)
    ulistDirect.append(u)
    vlistDirect.append(v)
    finalTime=state[0]
    listOfTrajOptim.append(list(listOfStateOptim))
    listOfTrajDirect.append(list(listOfStateDirect))
    finalTimes.append(finalTime)
    
meanTime=np.mean(finalTimes)
varTime=np.var(finalTimes)
print('Mean time steps to arrival : ' + str(meanTime))
print('Variance of time steps to arrival : ' + str(varTime))

#Mean time steps to arrival : 45.117 for UCT1.4 10 actions

#Mean time steps to arrival : 45.143 for UCT1.4 8 actions

#Mean time steps to arrival : 45.444 for UCT1.4 10 actions

#Mean time steps to arrival : 45.420 for UCT1.4 10 actions
#Variance of time steps to arrival : 1.875

#Mean time steps to arrival : 45.424 for UTC1.4 10 actions Continuous Directs
#Mean time steps to arrival : 45.753 for UTC1.4 14 actions
#Mean time steps to arrival : 45.900 for Direct Continuous
#Mean time steps to arrival : 45.936 for UTC10 14 actions
#Mean time steps to arrival : 46.058 for UTC1.4 14 actions Continuous Direct
#Mean time steps to arrival : 46.625 for Direct Discrete
#Mean time steps to arrival : 46.787 for UTC10 20 actions

#Mean time steps to arrival : 45.3761506276 for UCT1.4 8 actions continuous DP
#Mean time steps to arrival : 45.2958158996 for UCT1.4 8 actions continuous DP
#Mean time steps to arrival : 45.3163179916 for UCT1.4 8 actions continuous DP

 
# improve search with continuous heading for default policy only. 
# Sould compare with isochron method recomputed at each time step 
#%%
filehandler=open(name+'Results8Actions.pickle','wb')
pickle.dump([listOfTrajOptim,listOfTrajDirect,finalTimes,ulistOptim,vlistOptim,\
                                ulistDirect,vlistDirect],filehandler)
filehandler.close()


#%%
m = loaded.Simulator.praparePlotTraj2(loaded.rootNode.state,dl=0.3,dh=0.2)
loaded.Simulator.plotTraj(listOfStateOptim, m,color='red')
loaded.Simulator.plotTraj(listOfStateDirect, m)
x2, y2 = m(loaded.destination[1], loaded.destination[0])
m.scatter(x2,y2,color='red')
m.fillcontinents()                              
                                     

#%%
fg=plt.figure()
plt.hist(loaded.rewards,bins=50,range=[0,1])
plt.grid()
plt.ylabel('Number of rewards')
plt.xlabel('Value of rewards')
plt.title('max = ' + str(max(loaded.rewards))[:4]+', mean = ' + str(np.mean(loaded.rewards))[:4] +\
          ', var = ' + str(np.var(loaded.rewards))[:4])
fg.savefig('./ResultsBD/'+'rewards_'+name+'.pdf', bbox_inches='tight')


#%%
filehandler = open('BD_'+name+'.pickle', 'rb')
BDlist=pickle.load(filehandler)
filehandler.close()

for bd in BDlist : 
    bd.show()
#%%  
filehandler=open(name+'.pickle','rb')
loaded=pickle.load(filehandler)
filehandler.close()
listOfFigs=loaded.plotBD(3)
k=1
for fig in listOfFigs:
    ax=fig.gca()
    ax.set_xlim([-20,4])
    ax.set_ylim([-7,9])
    ax.grid()
    if k==3:
        loaded.plotBestChildren(loaded.rootNode,0,0,1,'red',ax)
    fig.savefig('./ResultsBD/latex'+str(k)+'BD_'+name+'.pdf', bbox_inches='tight')
    k=k+1
    fig.show()
#%%
fig=BDlist[-1]
ax=fig.gca()
loaded.plotBestChildren(loaded.rootNode,0,0,1,'red',ax)
fig.show()
#%%
fig.savefig('./ResultsBD/'+'BESTCHILD_'+name+'.pdf', bbox_inches='tight')

filehandler = open('BestChild'+name+'.pickle', 'wb')
pickle.dump(fig, filehandler)
filehandler.close()

##%%
#node=loaded.rootNode
##while node.children :
##    child=loaded.bestChild(node,0)
##    print(child)
##    node=child
#    
#fig=plt.figure()
#ax=plt.subplot(111)
#loaded.plotBestChildren(loaded.rootNode,0,0,1,'red',ax)

