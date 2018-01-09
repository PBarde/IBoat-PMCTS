#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import Hist
import matplotlib.pyplot as plt
from worker import UCT_COEFF
from math import log, sin, cos, pi
from operator import attrgetter

sys.path.append('../model/')
from simulatorTLKT import ACTIONS, Simulator, A_DICT

# number of scenarios
NUMSCENARIOS = 5


class MasterTree:
    """
    Master tree that manages ns different WorkerTree in parallel.
    Each WorkerTree is searching on a different Weather scenario.
    :ivar dict nodes: dictionary containing `MasterNode`, the keys are their corresponding hash
    :ivar numpy.array probability: array containing the probability of each scenario
    """

    def __init__(self):
        self.nodes = dict()
        self.probability = np.array([1 / NUMSCENARIOS for _ in range(NUMSCENARIOS)])
        self.nodes[hash(tuple([]))] = MasterNode(nodehash=hash(tuple([])))
        self.max_depth = None

    def integrate_buffer(self, buffer):
        """
        Integrates a list of update from a scenario. This method is to be called from a worker.
        :param buffer: list of updates coming from the worker. One update is a list :\
            [scenarioId, newNodeHash, parentHash, action, reward]
        :type buffer: list of list
        """
        for update in buffer:
            scenarioId, newNodeHash, parentHash, action, reward = update
            if newNodeHash not in self.nodes:
                self.nodes[newNodeHash] = MasterNode(newNodeHash, parentHash, action)

            self.nodes[newNodeHash].add_reward(scenarioId, reward)
            self.backup(newNodeHash, scenarioId, reward)

    def backup(self, nodehash, idscenario, reward):
        """
        Propagates the reward through the master tree.
        :param int nodehash: hash of the node
        :param int idscenario: id of the scenario/workertree where the update is coming
        :param float reward: reward of the update
        """
        node = self.nodes[nodehash]
        parent_hash = node.parentHash
        while parent_hash is not None:
            parent = self.nodes[parent_hash]
            parent.add_reward_action(idscenario, node.arm, reward)
            node = parent
            parent_hash = node.parentHash

    def update(self, worker_dict, event_dict, finish_event_dict):
        stop = False
        while not stop:
            for i, event in enumerate(event_dict.values()):
                # If a tree is ready)
                if event.isSet():
                    # Copy the buffer
                    buffer = worker_dict[i].copy_buffer()
                    # Clear the buffer
                    worker_dict[i].reset_buffer()
                    # Set the flag to false
                    event.clear()
                    # Add the new rewards in the master tree
                    self.integrate_buffer(buffer)

            # Test if all the workers are done
            if all(event.isSet() for event in finish_event_dict.values()):
                # End of the master thread
                stop = True

    def get_uct(self, worker_node):
        # warning here it is a reference toward a worker node.
        node_hash = hash(tuple(worker_node.origins))

        if node_hash not in self.nodes:
            print("Node " + str(node_hash) + " is not in the master")
            return 0

        else:
            # print("Node " + str(node_hash) + " is in the master")
            master_node = self.nodes[node_hash]
            uct_per_scenario = []
            for s, reward_per_scenario in enumerate(master_node.rewards):
                num_parent = 0
                uct_max_on_actions = 0

                for hist in self.nodes[master_node.parentHash].rewards[s]:
                    num_parent += sum(hist.h)

                if num_parent == 0:
                    uct_per_scenario.append(0)
                    continue

                num_node = sum(self.nodes[master_node.parentHash].rewards[s, A_DICT[master_node.arm]].h)

                exploration = UCT_COEFF * (2 * log(num_parent) / num_node) ** 0.5

                for hist in reward_per_scenario:
                    uct_value = hist.get_mean()

                    if uct_value > uct_max_on_actions:
                        uct_max_on_actions = uct_value

                uct_per_scenario.append(uct_max_on_actions + exploration)

            return np.dot(uct_per_scenario, self.probability)

    def get_children(self):
        nodes = dict(self.nodes)
        del nodes[hash(tuple([]))]  # remove the rootNode
        for node in nodes.values():
            self.nodes[node.parentHash].children.append(node)

    def get_depth(self):
        node = self.nodes[hash(tuple([]))]
        list_nodes = [node]
        node.depth = 0
        while list_nodes:
            node = list_nodes.pop(0)
            for n in node.children:
                list_nodes.append(n)
                n.depth = node.depth + 1

        # get max depth of the tree
        self.max_depth = max(map(lambda i: self.nodes[i].depth, self.nodes))

    def get_best_child(self, node):
        reward_per_action = np.zeros(shape=len(ACTIONS))
        temp = np.zeros(shape=NUMSCENARIOS)
        for j in range(len(ACTIONS)):
            for i in range(NUMSCENARIOS):
                temp[i] = node.rewards[i, j].get_mean()

            reward_per_action[j] = np.dot(temp, self.probability)
            print(np.dot(temp, self.probability))

        best_action = np.argmax(reward_per_action)
        print("best action" + str(best_action))
        for child in node.children:
            if A_DICT[child.arm] == best_action:
                return child

    def plot_tree(self):
        x0 = 0
        y0 = 0
        length = 1
        node = self.nodes[hash(tuple([]))]  # rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plot_children(node, x0, y0, length, ax, 'k')
        ax.scatter(0, 0, color='red', s=200, zorder=len(self.nodes))
        plt.axis('equal')
        fig.show()
        return fig, ax

    def plot_children(self, node, x, y, l, ax, color=None):
        x0 = x
        y0 = y
        for child in node.children:
            x = x0 + l * sin(child.arm * pi / 180)
            y = y0 + l * cos(child.arm * pi / 180)
            if color is None:
                col = str((child.depth / self.max_depth) * 0.8)
            else:
                col = color

            ax.plot([x0, x], [y0, y], color=col, marker='o', markersize='6')
            self.plot_children(child, x, y, l, ax, color=color)

    def plot_best_policy(self):
        fig, ax = self.plot_tree()
        node = self.nodes[hash(tuple([]))]  # rootNode
        x0 = 0
        y0 = 0
        length = 1
        for i in range(self.max_depth-1):
            node = self.get_best_child(node)
            x = x0 + length * sin(node.arm * pi / 180)
            y = y0 + length * cos(node.arm * pi / 180)
            ax.plot([x0, x], [y0, y], color="red", marker='o', markersize='6')
            x0 = x
            y0 = y
        return fig

    def plot_grey_tree(self):
        x0 = 0
        y0 = 0
        length = 1
        node = self.nodes[hash(tuple([]))]  # rootNode
        fig = plt.figure()
        ax = plt.subplot(111)
        self.plot_children(node, x0, y0, length, ax)
        ax.scatter(0, 0, color='red', s=200, zorder=len(self.nodes))
        plt.axis('equal')
        fig.show()
        return fig, ax

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


class MasterNode:
    """
    Node of a MasterTree
    :ivar int hash:
    :ivar int action:
    :ivar int parentHash:
    :ivar numpy.array rewards: Array of `Hist`
    :ivar list children: List of children (MasterNode)
    """

    def __init__(self, nodehash=None, parenthash=None, action=None):
        self.hash = nodehash
        self.arm = action
        self.parentHash = parenthash
        self.rewards = np.array([[Hist() for _ in range(len(ACTIONS))] for _ in range(NUMSCENARIOS)])
        self.children = []
        self.depth = None

    def add_reward(self, idscenario, reward):
        """
        Includes a reward into the histogram for all actions of one scenario.
        :param int idscenario: id of the scenario/workertree where the update is coming
        :param float reward: reward of the update
        """
        for hist in self.rewards[idscenario, :]:
            hist.add(reward)

    def add_reward_action(self, idscenario, action, reward):
        """
        Includes a reward into the histogram for one action of one scenario.
        :param idscenario:
        :param action:
        :param reward:
        :return:
        """
        self.rewards[idscenario, A_DICT[action]].add(reward)
