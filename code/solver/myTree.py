#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:06:46 2017

@author: paul
"""
import sys
sys.path.append("../model")
import matplotlib.pyplot as plt
from typing import List
import math
from math import exp,sqrt,asin
import simulatorTLKT as SimC
import random as rand
from timeit import default_timer as timer
import numpy as np
from utils import Hist

UCT_COEFF = 1 / 2**0.5
RHO=0.5
SEC_TO_DAYS = 1 / (60 * 60 * 24)


class Node:
    def __init__(self, state = None, parent = None, origins = [], \
                 children = [], depth = 0) :
        # the list() enables to copy the state in a new list and not just copy the reference
        self.state = tuple(state) # only for the rootNode
        self.parent = parent
        self.origins = list(origins)
        self.children = list(children)
        self.actions = list(SimC.ACTIONS)
        rand.shuffle(self.actions)
        self.Values = np.array([Hist() for i in SimC.ACTIONS])
        self.depth = depth


class Tree:
    def __init__(self, master, workerid, ite = 0, budget = 1000, simulator = None, \
                 destination = [], TimeMin=0):

        self.Master=master
        self.id=workerid
        self.ite = ite
        self.budget = budget
        self.Simulator = simulator
        self.destination = destination
        self.TimeMax = self.Simulator.times[-1]
        self.TimeMin = TimeMin
        self.depth = 0
        self.Nodes = []
        self.Buffer=[]

    def uct_search(self, rootState):

        # We create the root node and add it to the tree
        rootNode = Node(state=rootState)
        self.rootNode = rootNode
        self.Nodes.append(rootNode)
        #        print(Node.getHash(rootState))

        # while we still have computationnal budget we expand nodes
        while self.ite < self.budget:
            # the treePolicy gives us the reference to the newly expanded node
            startTreePolicy = timer()

            leafNode = self.treePolicy(self.rootNode)

            endTreePolicy = timer()
            timeTreePolicy = endTreePolicy - startTreePolicy
            #            print('Elapsed time Tree policy= ' + str(timeTreePolicy))

            startDefaultPolicy = timer()

            leafNode.R = self.defaultPolicy(leafNode)

            endDefaultPolicy = timer()
            timeDefaultPolicy = endDefaultPolicy - startDefaultPolicy
            #            print(', Elapsed time Default policy= ' + str(timeDefaultPolicy))

            startBackUp = timer()
            self.backUp(leafNode, [leafNode.R, leafNode.N])
            endBackUp = timer()

            timeBackUp = endBackUp - startBackUp
            #            print(', Elapsed time BackUp= ' + str(timeBackUp) + '\n')

            totalETime = endBackUp - startTreePolicy

            self.ite = self.ite + 1
            print('\n Iteration ' + str(self.ite) + 'on ' + str(self.budget) + ' : \n')
            print('Tree Policy = ' + str(timeTreePolicy / totalETime) + ', Default Policy = ' \
                  + str(timeDefaultPolicy / totalETime) + ', Time Backup = ' + \
                  str(timeBackUp / totalETime) + '\n')

    def treePolicy(self, node):

        while not self.isNodeTerminal(node):

            if not self.isFullyExpanded(node):
                return self.expand(node)

            else:
                node = self.bestChild(node)

        return node

    def expand(self, node):
        action = node.actions.pop()
        newNode = Node(parent=node, origins=node.origins+action, depth=node.depth + 1)
        self.depth = max(self.depth, newNode.depth)
        node.children.append(newNode)
        self.Nodes.append(newNode)
        return newNode
    
#TODO a modif
    def bestChild(self, node):
        UTCs = []
        for child in node.children:
            UTCs.append(self.getUCT(child))
        max_UTC = max(UTCs)
        max_index = UTCs.index(max_UTC)
        return node.children[max_index]

    def getUCT(self, node):
        return node.R / node.N + UCT_COEFF * (2 * math.log(node.parent.N) / node.N) ** 0.5

    def defaultPolicy(self, node):

        self.getSimToEstimateState(node)

        dist, action = self.Simulator.getDistAndBearing(self.Simulator.state[1:],self.destination)
        atDest,frac =Tree.isStateAtDest(self.destination,self.Simulator.prevState,self.Simulator.state)
              
        while (not atDest) \
                and (not Tree.isStateTerminal(self.Simulator,self.Simulator.state)):
            self.Simulator.doStep(action)
            dist, action = self.Simulator.getDistAndBearing(self.Simulator.state[1:],self.destination)
            atDest,frac =Tree.isStateAtDest(self.destination,self.Simulator.prevState,self.Simulator.state)

        if atDest:
            finalTime = self.Simulator.times[self.Simulator.state[0]]-(1-frac)
            reward = (exp((self.TimeMax*1.001 - finalTime) / (self.TimeMax*1.001 - self.TimeMin)) - 1) / (exp(1) - 1)
#            reward=(self.TimeMax*1.001 - finalTime) / (self.TimeMax*1.001 - self.TimeMin)
        else:
            reward = 0
            finalTime = self.TimeMax

        print('Final dist = ' + str(dist) + ', final Time = ' + str(finalTime) + \
              ', reward = ' + str(reward))
        self.rewards.append(reward)
        return reward

    def getSimToEstimateState(self, node):
        listOfActions = []

        while node is not self.rootNode:
            listOfActions.append(node.origin)
            node = node.parent

        self.Simulator.reset(self.rootNode.state)

        while listOfActions and not Tree.isStateTerminal(self.Simulator,self.Simulator.state)\
                            and not Tree.isStateAtDest(self.destination,self.Simulator.prevState,self.Simulator.state) :
            action = listOfActions.pop()
            self.Simulator.doStep(action)
            
    @staticmethod
    def isStateAtDest(destination, stateA, stateB):
        [xa,ya,za]=SimC.Simulator.fromGeoToCartesian(stateA[1:])
        [xb,yb,zb]=SimC.Simulator.fromGeoToCartesian(stateB[1:])
        [xd,yd,zd]=SimC.Simulator.fromGeoToCartesian(destination)
        c=(yb/ya*xa-xb)/(zb-yb/ya*za)
        b=-(xa+c*za)/ya
        d=abs(xd+b*yd+c*zd)/sqrt(1+b**2+c**2)
        alpha=asin(d)
        
        if alpha>SimC.DESTINATION_ANGLE : return [False,None]
        
        else : 
          vad=np.array([xd,yd,zd])-np.array([xa,ya,za])
          vdb=np.array([xb,yb,zb])-np.array([xd,yd,zd])
          vab=np.array([xb,yb,zb])-np.array([xa,ya,za])
          
          p=np.dot(vad,vdb)
          
          if p<0 : return [False,None]
      
          else : return [True,np.dot(vad,vab)/np.dot(vab,vab)]
    
    @staticmethod
    def isStateTerminal(simulator, state):
        if simulator.times[state[0]] == simulator.times[-1]:
            return True

        elif state[1] < simulator.lats[0] or state[1] > simulator.lats[-1]:
            return True

        elif state[2] < simulator.lons[0] or state[2] > simulator.lons[-1]:
            return True
        else:
            return False

    def isNodeTerminal(self, node):
        return self.Simulator.times[node.depth] == self.TimeMax

    def isFullyExpanded(self, node):
        return len(node.actions) == 0

    def backUp(self, node: Node, Q) -> None:
        while node.parent:
            node.parent.R = node.parent.R + Q[0]
            node.parent.N = node.parent.N + Q[1]
            node = node.parent

    def plotTree(self) :
        x0 = 0
        y0 = 0
        l = 1
        node = self.rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plotChildren(node, x0, y0, l,'k', ax)
        ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        plt.axis('equal')
        fig.show()
        return fig

    def plotGreyTree(self):
        x0 = 0
        y0 = 0
        l = 1
        node = self.rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plotChildren(node, x0, y0, l,'0', ax)
        ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        plt.axis('equal')
        fig.show()
        return fig
        
    def plotChildren(self, node, x, y, l, ax):
        x0 = x
        y0 = y
        for child in node.children:
            x = x0 + l * math.sin(child.origin * math.pi / 180)
            y = y0 + l * math.cos(child.origin * math.pi / 180)
            color=str((child.depth/self.depth)*0.8)
            ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')
            self.plotChildren(child, x, y, l,color, ax)
            
    def plotBestChildren(self, node, x, y, l, color, ax):
        x0 = x
        y0 = y
        
        while node.children :
            child=self.bestChild(node,0)
            print(child)
            x = x0 + l * math.sin(child.origin * math.pi / 180)
            y = y0 + l * math.cos(child.origin * math.pi / 180)
            ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')
            x0=x
            y0=y
            node=child
            
    def plotChildrenBD(self, node,nodes, x, y, l, ax):
        x0 = x
        y0 = y
        for child in node.children:
            if child in nodes: 
                x = x0 + l * math.sin(child.origin * math.pi / 180)
                y = y0 + l * math.cos(child.origin * math.pi / 180)
                color=str((child.depth/self.depth)*0.8)
                ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')
                
            else: break
            self.plotChildrenBD(child, nodes, x, y, l,color, ax)
            
    def plotBD(self, nBD=2):
        Nnodes=len(self.Nodes)
        Dnodes=int(Nnodes/nBD)
        listOfFig=[]
        listOfAx=[]
        x0 = 0
        y0 = 0
        l = 1
        
        for n in range(nBD):
            
            fig = plt.figure()   
            listOfFig.append(fig)
            ax = plt.subplot(111)
            listOfAx.append(ax)
            nodes= self.Nodes[0:(n + 1) * Dnodes]
            self.plotChildrenBD(self.rootNode,nodes,x0,y0,l,'0',ax)
            ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        return listOfFig
