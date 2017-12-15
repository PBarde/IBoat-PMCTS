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
    """

    def __init__(self, state=[]):
        self.nodes = dict()
        self.probability = np.array([1 / NUMSCENARIOS for _ in range(NUMSCENARIOS)])
        self.rootState = state
        self.nodes[0] = MasterNode(nodehash=hash(0))

    def integrate_buffer(self, buffer):
        for update in buffer:
            scenarioId, newNodeHash, parentHash, action, reward = update
            if newNodeHash not in self.nodes:
                self.nodes[newNodeHash] = MasterNode(newNodeHash, parentHash, action)

            self.nodes[newNodeHash].add_reward(scenarioId, reward)
            self.backup(newNodeHash, scenarioId, reward)

    def backup(self, nodehash, idscenario, reward):
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
    """

    def __init__(self, nodehash=None, parenthash=None, action=None):
        self.hash = nodehash
        self.arm = action
        self.parentHash = parenthash
        self.rewards = np.array([[Hist() for _ in range(len(ACTIONS))] for _ in range(NUMSCENARIOS)])

    def add_reward(self, idscenario, reward):
        for hist in self.rewards[idscenario, :]:
            hist.add(reward)

    def add_reward_action(self, idscenario, action, reward):
        self.rewards[idscenario, action].add(reward)
