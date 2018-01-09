#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:06:46 2017

@author: paul
"""
import sys

sys.path.append("../model/")
import matplotlib.pyplot as plt
import math
from math import exp, sqrt, asin
import simulatorTLKT as SimC
import random as rand
from timeit import default_timer as timer
import numpy as np
from utils import Hist
from time import sleep

UCT_COEFF = 1 / 2 ** 0.5
RHO = 0.5
SEC_TO_DAYS = 1 / (60 * 60 * 24)


class Node:
    def __init__(self, state=None, parent=None, origins=[], children=[], depth=0):
        # the list() enables to copy the state in a new list and not just copy the reference
        if state is not None:
            self.state = tuple(state)  # only for the rootNode
        else:
            self.state = None
        self.parent = parent
        self.origins = list(origins)
        self.children = list(children)
        self.actions = list(SimC.ACTIONS)
        rand.shuffle(self.actions)
        self.Values = np.array([Hist() for i in SimC.ACTIONS])
        self.depth = depth


class Tree:
    def __init__(self, master, workerid, ite=0, budget=1000, simulator=None, destination=[], TimeMin=0):

        self.Master = master
        self.id = workerid
        self.ite = ite
        self.budget = budget
        self.Simulator = simulator
        self.destination = destination
        self.TimeMax = self.Simulator.times[-1]
        self.TimeMin = TimeMin
        self.depth = 0
        self.Nodes = []
        self.Buffer = []

    def uct_search(self, rootState, frequency, event, end_event):
        # We create the root node and add it to the tree
        rootNode = Node(state=rootState)
        self.rootNode = rootNode
        self.Nodes.append(rootNode)
        #        print(Node.getHash(rootState))
        self.event = event
        self.end_event = end_event
        # while we still have computationnal budget we expand nodes
        while self.ite < self.budget:
            # the treePolicy gives us the reference to the newly expanded node
            startTreePolicy = timer()

            leafNode = self.tree_policy(self.rootNode)

            endTreePolicy = timer()
            timeTreePolicy = endTreePolicy - startTreePolicy
            #            print('Elapsed time Tree policy= ' + str(timeTreePolicy))

            startDefaultPolicy = timer()

            reward = self.default_policy(leafNode)

            endDefaultPolicy = timer()
            timeDefaultPolicy = endDefaultPolicy - startDefaultPolicy
            #            print(', Elapsed time Default policy= ' + str(timeDefaultPolicy))

            startBackUp = timer()
            Tree.back_up(leafNode, reward)
            endBackUp = timer()

            timeBackUp = endBackUp - startBackUp
            #            print(', Elapsed time BackUp= ' + str(timeBackUp) + '\n')

            totalETime = endBackUp - startTreePolicy
            self.Buffer.append(
                [self.id, hash(tuple(leafNode.origins)), hash(tuple(leafNode.origins[:-1])), leafNode.origins[-1],
                 reward])
            self.ite = self.ite + 1

            print(
                '\n Iteration ' + str(self.ite) + ' on ' + str(self.budget) + ' for workers ' + str(self.id) + ' : \n')
            print('Tree Policy = ' + str(timeTreePolicy / totalETime) + ', Default Policy = ' \
                  + str(timeDefaultPolicy / totalETime) + ', Time Backup = ' + \
                  str(timeBackUp / totalETime) + '\n')

            # Notify the master that the buffer is ready
            if self.ite % frequency == 0:
                event.set()
                # wait for the master to reset the buffer
                while event.isSet():
                    print("waiting, event is set :" + str(event.isSet()) + "for worker num " + str(self.id))

        # Set the end_event to True to notify the master that the search is done
        end_event.set()

    def reset_buffer(self):
        self.Buffer = []

    def copy_buffer(self):
        return list(self.Buffer)

    def tree_policy(self, node):

        while not self.is_node_terminal(node):

            if not Tree.is_fully_expanded(node):
                return self.expand(node)

            else:
                node = self.best_child(node)

        return node

    def expand(self, node):
        action = node.actions.pop()
        newNode = Node(parent=node, origins=node.origins + [action], depth=node.depth + 1)
        self.depth = max(self.depth, newNode.depth)
        node.children.append(newNode)
        self.Nodes.append(newNode)
        return newNode

    def best_child(self, node):
        max_ucts_of_children = 0
        id_of_best_child = 0
        num_node = 0

        for val in node.Values:
            num_node += sum(val.h)

        for i, child in enumerate(node.children):
            uct_master = self.Master.get_uct(child)
            if uct_master == 0:
                ucts_of_children = Tree.get_uct(child, num_node)
            else:
                ucts_of_children = (1 - RHO) * Tree.get_uct(child, num_node) + RHO * uct_master

            if ucts_of_children > max_ucts_of_children:
                max_ucts_of_children = ucts_of_children
                id_of_best_child = i

        return node.children[id_of_best_child]

    @staticmethod
    def get_uct(node, num_parent):
        uct_max_on_actions = 0
        ii = SimC.A_DICT[node.origins[-1]]
        num_node = sum(node.parent.Values[ii].h)
        exploration = UCT_COEFF * (2 * math.log(num_parent) / num_node) ** 0.5

        for hist in node.Values:
            uct_value = hist.get_mean()

            if uct_value > uct_max_on_actions:
                uct_max_on_actions = uct_value

        return uct_max_on_actions + exploration

    def default_policy(self, node):

        self.get_sim_to_estimate_state(node)
        dist, action = self.Simulator.getDistAndBearing(self.Simulator.state[1:], self.destination)
        atDest, frac = Tree.is_state_at_dest(self.destination, self.Simulator.prevState, self.Simulator.state)

        while (not atDest) \
                and (not Tree.is_state_terminal(self.Simulator, self.Simulator.state)):
            self.Simulator.doStep(action)
            dist, action = self.Simulator.getDistAndBearing(self.Simulator.state[1:], self.destination)
            atDest, frac = Tree.is_state_at_dest(self.destination, self.Simulator.prevState, self.Simulator.state)

        if atDest:
            finalTime = self.Simulator.times[self.Simulator.state[0]] - (1 - frac)
            reward = (exp((self.TimeMax * 1.001 - finalTime) / (self.TimeMax * 1.001 - self.TimeMin)) - 1) / (
                exp(1) - 1)
        else:
            reward = 0
            finalTime = self.TimeMax

        print('Final dist = ' + str(dist) + ', final Time = ' + str(finalTime) + \
              ', reward = ' + str(reward))
        return reward

    def get_sim_to_estimate_state(self, node):
        listOfActions = list(node.origins)
        listOfActions.reverse()
        self.Simulator.reset(self.rootNode.state)

        action = listOfActions.pop()
        self.Simulator.doStep(action)

        while listOfActions and not Tree.is_state_terminal(self.Simulator, self.Simulator.state) \
                and not Tree.is_state_at_dest(self.destination, self.Simulator.prevState, self.Simulator.state):
            action = listOfActions.pop()
            self.Simulator.doStep(action)

    @staticmethod
    def is_state_at_dest(destination, stateA, stateB):
        [xa, ya, za] = SimC.Simulator.fromGeoToCartesian(stateA[1:])
        [xb, yb, zb] = SimC.Simulator.fromGeoToCartesian(stateB[1:])
        [xd, yd, zd] = SimC.Simulator.fromGeoToCartesian(destination)
        c = (yb / ya * xa - xb) / (zb - yb / ya * za)
        b = -(xa + c * za) / ya
        d = abs(xd + b * yd + c * zd) / sqrt(1 + b ** 2 + c ** 2)
        alpha = asin(d)

        if alpha > SimC.DESTINATION_ANGLE:
            return [False, None]

        else:
            vad = np.array([xd, yd, zd]) - np.array([xa, ya, za])
            vdb = np.array([xb, yb, zb]) - np.array([xd, yd, zd])
            vab = np.array([xb, yb, zb]) - np.array([xa, ya, za])

            p = np.dot(vad, vdb)

            if p < 0:
                return [False, None]

            else:
                return [True, np.dot(vad, vab) / np.dot(vab, vab)]

    @staticmethod
    def is_state_terminal(simulator, state):
        if simulator.times[state[0]] == simulator.times[-1]:
            return True

        elif state[1] < simulator.lats[0] or state[1] > simulator.lats[-1]:
            return True

        elif state[2] < simulator.lons[0] or state[2] > simulator.lons[-1]:
            return True
        else:
            return False

    def is_node_terminal(self, node):
        return self.Simulator.times[node.depth] == self.TimeMax

    @staticmethod
    def is_fully_expanded(node):
        return len(node.actions) == 0

    @staticmethod
    def back_up(node, Q):
        # the first reward of a node is shared by all the actions
        for hist in node.Values:
            hist.add(Q)
        # then the reward is propagated to the parent node according to the action that expanded the
        # child
        while node.parent:
            ii = SimC.A_DICT[node.origins[-1]]
            node.parent.Values[ii].add(Q)
            node = node.parent

    def plot_tree(self):
        x0 = 0
        y0 = 0
        l = 1
        node = self.rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plot_children(node, x0, y0, l, 'k', ax)
        ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        plt.axis('equal')
        fig.show()
        return fig

    def plot_grey_tree(self):
        x0 = 0
        y0 = 0
        l = 1
        node = self.rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plot_children(node, x0, y0, l, '0', ax)
        ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        plt.axis('equal')
        fig.show()
        return fig

    def plot_children(self, node, x, y, l, ax):
        x0 = x
        y0 = y
        for child in node.children:
            x = x0 + l * math.sin(child.origin * math.pi / 180)
            y = y0 + l * math.cos(child.origin * math.pi / 180)
            color = str((child.depth / self.depth) * 0.8)
            ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')
            self.plot_children(child, x, y, l, color, ax)

    def plot_best_children(self, node, x, y, l, color, ax):
        x0 = x
        y0 = y

        while node.children:
            child = self.best_child(node, 0)
            print(child)
            x = x0 + l * math.sin(child.origin * math.pi / 180)
            y = y0 + l * math.cos(child.origin * math.pi / 180)
            ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')
            x0 = x
            y0 = y
            node = child

    def plot_children_bd(self, node, nodes, x, y, l, ax):
        x0 = x
        y0 = y
        for child in node.children:
            if child in nodes:
                x = x0 + l * math.sin(child.origin * math.pi / 180)
                y = y0 + l * math.cos(child.origin * math.pi / 180)
                color = str((child.depth / self.depth) * 0.8)
                ax.plot([x0, x], [y0, y], color=color, marker='o', markersize='6')

            else:
                break
            self.plot_children_bd(child, nodes, x, y, l, color, ax)

    def plot_bd(self, nBD=2):
        Nnodes = len(self.Nodes)
        Dnodes = int(Nnodes / nBD)
        listOfFig = []
        listOfAx = []
        x0 = 0
        y0 = 0
        l = 1

        for n in range(nBD):
            fig = plt.figure()
            listOfFig.append(fig)
            ax = plt.subplot(111)
            listOfAx.append(ax)
            nodes = self.Nodes[0:(n + 1) * Dnodes]
            self.plot_children_bd(self.rootNode, nodes, x0, y0, l, '0', ax)
            ax.scatter(0, 0, color='red', s=200, zorder=self.ite)
        return listOfFig
