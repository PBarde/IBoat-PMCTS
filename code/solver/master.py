#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import Hist

sys.path.append('../model')
from simulatorTLKT import ACTIONS, Simulator

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
        self.nodes[0] = MasterNode(nodehash=hash(0))

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
        parentHash = node.parentHash
        while parentHash is not None:
            parent = self.nodes[parentHash]
            parent.add_reward_action(idscenario, node.arm, reward)
            node = parent
            parentHash = node.parentHash


class MasterNode:
    """
    Node of a MasterTree
    :ivar int hash:
    :ivar int action:
    :ivar int parentHash:
    :ivar numpy.array rewards: Array of `Hist`
    """

    def __init__(self, nodehash=None, parenthash=None, action=None):
        self.hash = nodehash
        self.arm = action
        self.parentHash = parenthash
        self.rewards = np.array([[Hist() for _ in range(len(ACTIONS))] for _ in range(NUMSCENARIOS)])

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
        self.rewards[idscenario, action].add(reward)
