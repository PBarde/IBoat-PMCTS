#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:06:46 2017

@author: paul
"""
import sys
import math
import random
from math import exp, sqrt, asin, log
import random as rand
import numpy as np
from utils import Hist
from master import MasterNode

sys.path.append("../model/")
import simulatorTLKT as SimC
from simulatorTLKT import A_DICT, ACTIONS

UCT_COEFF = 1 / 2 ** 0.5
RHO = 0.5


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

    def back_up(self, reward):
        # the first reward of a node is shared by all the actions
        for hist in self.Values:
            hist.add(reward)
        # then the reward is propagated to the parent node according to the action that expanded the
        # child
        node = self
        while node.parent:
            ii = SimC.A_DICT[node.origins[-1]]
            node.parent.Values[ii].add(reward)
            node = node.parent

    def is_fully_expanded(self):
        return len(self.actions) == 0

    def get_uct(self, num_parent):
        uct_max_on_actions = 0
        ii = SimC.A_DICT[self.origins[-1]]
        num_node = sum(self.parent.Values[ii].h)
        exploration = UCT_COEFF * (2 * math.log(num_parent) / num_node) ** 0.5

        for hist in self.Values:
            uct_value = hist.get_mean()

            if uct_value > uct_max_on_actions:
                uct_max_on_actions = uct_value

        return uct_max_on_actions + exploration


class Tree:
    def __init__(self, workerid, nscenario, probability=[], ite=0, budget=1000,
                 simulator=None, destination=[], TimeMin=0, buffer=[]):
        self.id = workerid
        self.ite = ite
        self.budget = budget
        self.Simulator = simulator
        self.destination = destination
        self.TimeMax = self.Simulator.times[-1]
        self.TimeMin = TimeMin
        self.depth = 0
        self.Nodes = []
        self.buffer = buffer
        self.numScenarios = nscenario
        if len(probability) == 0:
            self.probability = np.array([1 / nscenario for _ in range(nscenario)])
        else:
            self.probability = probability

    def uct_search(self, rootState, frequency, Master_nodes):
        # We create the root node and add it to the tree
        rootNode = Node(state=rootState)
        self.rootNode = rootNode
        self.Nodes.append(rootNode)

        # While we still have computational budget we expand nodes
        while self.ite < self.budget:
            # the treePolicy gives us the reference to the newly expanded node
            leafNode = self.tree_policy(self.rootNode, Master_nodes)

            # The default policy gives the reward
            reward = self.default_policy(leafNode)

            # Propagate the reward through the tree
            leafNode.back_up(reward)

            # Update the buffer
            self.buffer.append(
                [self.id, hash(tuple(leafNode.origins)), hash(tuple(leafNode.origins[:-1])), leafNode.origins[-1],
                 reward])
            self.ite = self.ite + 1

            # Print every 50 ite
            if self.ite % 50 == 0: print(
                '\n Iteration ' + str(self.ite) + ' on ' + str(self.budget) + ' for workers ' + str(self.id) + ' : \n')

            # Notify the master that the buffer is ready
            if self.ite % frequency == 0:
                self.integrate_buffer(Master_nodes)
                self.buffer = []

    def tree_policy(self, node, master_nodes):
        while not self.is_node_terminal(node):
            if (random.random() < 0.5) and node.children:
                node = self.best_child(node, master_nodes)
            elif not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node, master_nodes)
        return node

    def expand(self, node):
        action = node.actions.pop()
        newNode = Node(parent=node, origins=node.origins + [action], depth=node.depth + 1)
        self.depth = max(self.depth, newNode.depth)
        node.children.append(newNode)
        self.Nodes.append(newNode)
        return newNode

    def best_child(self, node, Master_nodes):
        max_ucts_of_children = -1
        id_of_best_child = -1
        num_node = 0

        for val in node.Values:
            num_node += sum(val.h)

        for i, child in enumerate(node.children):
            uct_master = self.get_master_uct(hash(tuple(child.origins)), Master_nodes)
            if uct_master == 0:
                ucts_of_children = child.get_uct(num_node)
            else:
                ucts_of_children = (1 - RHO) * child.get_uct(num_node) + RHO * uct_master

            if ucts_of_children > max_ucts_of_children:
                max_ucts_of_children = ucts_of_children
                id_of_best_child = i
        return node.children[id_of_best_child]

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
            finalTime = self.Simulator.times[self.Simulator.state[0]] - (1 - frac) * (
                self.Simulator.times[self.Simulator.state[0]] - self.Simulator.times[self.Simulator.state[0] - 1])
            reward = (exp((self.TimeMax * 1.001 - finalTime) / (self.TimeMax * 1.001 - self.TimeMin)) - 1) / (
                exp(1) - 1)
        else:
            reward = 0

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

    def get_master_uct(self, node_hash, Master_nodes):
        """
        Compute the master uct value of a worker node.

        :param int node_hash: the corresponding hash node
        :return float: The uct value of the worker node passed in parameter
        """
        master_node = Master_nodes.get(node_hash, 0)
        if master_node == 0:
            # print("Node " + str(node_hash) + " is not in the master")
            return 0

        else:
            # print("Node " + str(node_hash) + " is in the master")
            uct_per_scenario = []
            for s, reward_per_scenario in enumerate(master_node.rewards):
                num_parent = 0
                uct_max_on_actions = 0
                for hist in master_node.parentNode.rewards[s]:
                    num_parent += sum(hist.h)

                num_node = sum(master_node.parentNode.rewards[s, A_DICT[master_node.arm]].h)

                if (num_parent == 0) or (num_node == 0):
                    uct_per_scenario.append(0)
                    continue

                exploration = UCT_COEFF * (2 * log(num_parent) / num_node) ** 0.5

                for hist in reward_per_scenario:
                    uct_value = hist.get_mean()

                    if uct_value > uct_max_on_actions:
                        uct_max_on_actions = uct_value

                uct_per_scenario.append(uct_max_on_actions + exploration)

            return np.dot(uct_per_scenario, self.probability)

    def integrate_buffer(self, Master_nodes):
        """
        Integrates a list of update from a scenario. This method is to be called from a worker.

        :param buffer: list of updates coming from the worker. One update is a list :\
            [scenarioId, newNodeHash, parentHash, action, reward]
        :type buffer: list of list
        """

        for update in self.buffer:
            # print(update)
            scenarioId, newNodeHash, parentHash, action, reward = update
            node = Master_nodes.get(newNodeHash, 0)

            if node == 0:
                node = MasterNode(self.numScenarios, nodehash=newNodeHash,
                                  parentNode=Master_nodes[parentHash], action=action)

            node.add_reward(scenarioId, reward)

            # todo faire pareil dans le backup !
            Master_nodes[newNodeHash] = node
            while node.parentNode is not None:
                parent_hash = Master_nodes[node.hash].parentNode.hash
                parent_node = Master_nodes[parent_hash]
                parent_node.add_reward_action(scenarioId, node.arm, reward)
                # parent_node.children.append(node)
                Master_nodes[parent_hash] = parent_node
                node = parent_node

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
