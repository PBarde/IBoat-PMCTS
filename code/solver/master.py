#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from utils import Hist

sys.path.append('../model/')
from simulatorTLKT import ACTIONS, Simulator, A_DICT
from worker import UCT_COEFF
from math import log
from time import sleep

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
            # print("Node " + str(node_hash) + " is not in the master")
            return 0

        else:
            print("Node " + str(node_hash) + " is in the master")
            master_node = self.nodes[node_hash]
            uct_per_scenario = []
            for s, reward_per_scenario in enumerate(master_node.rewards):
                num_parent = 0
                uct_max_on_actions = 0

                for hist in self.nodes[master_node.parentHash].rewards[s]:
                    num_parent += sum(hist.h)

                num_node = sum(self.nodes[master_node.parentHash].rewards[s, A_DICT[master_node.arm]].h)
                exploration = UCT_COEFF * (2 * log(num_parent) / num_node) ** 0.5

                for hist in reward_per_scenario:
                    uct_value = hist.get_mean()

                    if uct_value > uct_max_on_actions:
                        uct_max_on_actions = uct_value

                uct_per_scenario.append(uct_max_on_actions + exploration)

            return np.dot(uct_per_scenario, self.probability)


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
        self.rewards[idscenario, A_DICT[action]].add(reward)
