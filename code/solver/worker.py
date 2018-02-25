#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:06:46 2017

@author: paul
"""
import sys
import math
from math import exp, sqrt, asin, log
import random as rand
import numpy as np
from utils import Hist
from master_node import MasterNode

sys.path.append("../model/")
import simulatorTLKT as SimC
from simulatorTLKT import A_DICT, ACTIONS

UCT_COEFF = 1 / 5 * 1 / 2 ** 0.5
RHO = 0.5


class Node:
    """
    Node of a :class:`worker.Tree`.

    :ivar tuple state: Only for the root Node: initial state (time, lat, lon), None for other node.
    :ivar worker.Node parent: Reference to the parent of this node.
    :ivar list origins: sequel of actions taken from to root node to this node.
    :ivar list children: Child nodes of this node.
    :ivar list actions: Remaining actions available (not expanded) from this node in random order.
    :ivar numpy.array Values: Array of :class:`utils.Hist` to save the rewards. Its size is the number of possible actions.
    :ivar int depth: Depth of the node in the Tree.

    """

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
        self.Values = np.array([Hist() for _ in SimC.ACTIONS])
        self.depth = depth

    def back_up(self, reward):
        """
        Propagates the reward through the Tree, starting from this :class:`worker.Node`, up to the root.

        :param float reward: The reward corresponding to the expansion of this :class:`worker.Node`.
        """
        # the first reward of a node is put in a random action
        action = np.random.randint(len(ACTIONS))
        self.Values[action].add(reward)

        # then the reward is propagated to the parent node according to the action that expanded the child
        node = self
        while node.parent:
            ii = SimC.A_DICT[node.origins[-1]]
            node.parent.Values[ii].add(reward)
            node = node.parent

    def is_fully_expanded(self):
        """

        Returns True if this :class:`worker.Node` is fully expanded (if there is not remaining actions)

        :return boolean: True if fully expanded, False otherwise.

        """
        return len(self.actions) == 0

    def get_uct(self, num_parent):
        """
        Computes the uct values of this :class:`worker.Node` (combination of exploration and exploitation)

        :param int num_parent: Number of times the parent of the :class:`worker.Node` has been explored.

        :return float: The uct value.

        """
        uct_max_on_actions = 0
        num_node = 0
        for val in self.Values:
            num_node += sum(val.h)

        exploration = UCT_COEFF * (2 * math.log(num_parent) / num_node) ** 0.5

        for hist in self.Values:
            uct_value = hist.get_mean()

            if uct_value > uct_max_on_actions:
                uct_max_on_actions = uct_value

        return uct_max_on_actions + exploration


class Tree:
    """

    A tree which represents a MCTS on one weather scenario.

    :ivar int id: Id of the tree, corresponding to the id scenario.
    :ivar int ite: Current number of iterations done.
    :ivar int budget: Max. number of iterations
    :ivar Simulator simulator: The simulator used to do the boat simulations (step, etc.).
    :ivar list destination: Position [lat, lon] of the wanted destination.
    :ivar float TimeMax: Time horizon of the search.
    :ivar float TimeMin: Minimum time to arrive to the destination, computed on several boats which go straight \
    from the initial point to the destination (default policy).
    :ivar int depth: Maximum depth of the tree.
    :ivar list Nodes: List of :class:`worker.Node`, representing the tree.
    :ivar list buffer: The buffer is a list of updates to be included in the master Tree. \
    One update is a list : [scenarioId, newNodeHash, parentHash, action, reward].
    :ivar int numScenarios: Number total of scenarios used during the MCT parallel search.
    :ivar numpy.array probability: array containing the probability of each scenario. Scenario id as a
                                    probability probability[id] to occur.
    """

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
        """
        Launches the MCTS for the scenario.

        :param list rootState: State [t_index, lat, lon] of the root node.
        :param int frequency: Length of the buffer: number of iterations between each buffer integrations.
        :param dict Master_nodes: `Manager <https://docs.python.org/2/library/multiprocessing.html#sharing-state-\
        between-processes>`_ (Dictionary of :class:`master_node.MasterNode`) which saves the nodes of every scenario.

        """
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
        """
        Method implementing the tree policy phase in the MCTS-UCT search. Starts from the root nodes and
        iteratively selects the best child of the nodes. A node must be fully expanded before we continue down
        to its best child.

        :param node: starting node of the tree policy, usually the root node.
        :param dict Master_nodes: `Manager <https://docs.python.org/2/library/multiprocessing.html#sharing-state-\
        between-processes>`_ (Dictionary of :class:`master_node.MasterNode`) which saves the nodes of every scenario.
        :return: the newly expanded node
        :rtype: :class:`Node`
        """
        while not self.is_node_terminal(node):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node, master_nodes)
        return node

    def expand(self, node):
        """

        Creates a new :class:`worker.Node` from a node (its parent). The new node is expanded randomly \
        using an available actions.

        :param worker.Node node: The parent node.
        :return: The new :class:`worker.Node`.

        """
        action = node.actions.pop()
        newNode = Node(parent=node, origins=node.origins + [action], depth=node.depth + 1)
        self.depth = max(self.depth, newNode.depth)
        node.children.append(newNode)
        self.Nodes.append(newNode)
        return newNode

    def best_child(self, node, Master_nodes):
        """
        Select the best child of a node, by comparing their uct values. The comparison is based on the value of the \
        child in this tree, but also in the master tree, if it exists there.

        :param `worker.Node` node: The parent node.
        :param dict Master_nodes: `Manager <https://docs.python.org/2/library/multiprocessing.html#sharing-state-\
        between-processes>`_ (Dictionary of :class:`master_node.MasterNode`) which saves the nodes of every scenario.

        :return: The best child (:class:`worker.Node`) of the parent node given in parameter.
        """

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
        """
        Policy used to compute the reward of a node. First, the state of the node is estimated with \
        :meth:`get_sim_to_estimate_state`, then the default policy is applied (going straight to the destination).

        :param worker.Node node: The node one wants to evaluate.
        :return float: The rewards of the node.
        """
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
        """
        Brings the simulator to an estimate state (time, lat, lon) of a node. Since the dynamic is not deterministic, \
        the state is an estimation.

        :param worker.Node node: The nodes one wants to estimate.
        """
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
        Compute the uct value seen by the master tree.

        :param int node_hash: the corresponding hash node.
        :param dict Master_nodes: `Manager <https://docs.python.org/2/library/multiprocessing.html#sharing-state-\
        between-processes>`_ (Dictionary of :class:`master_node.MasterNode`) which saves the nodes of every scenario.
        :return float: The uct value of the node passed in parameter.
        """
        master_node = Master_nodes.get(node_hash, 0)
        idx_scenarios = []

        # If the node is not in the the master dictionary, the uct is 0
        if master_node == 0:
            return 0

        else:
            uct_per_scenario = []
            for s, reward_per_scenario in enumerate(master_node.rewards):
                uct_max_on_actions = 0

                num_parent = 0
                for hist in master_node.parentNode.rewards[s]:
                    num_parent += sum(hist.h)

                num_node = 0
                for hist in master_node.rewards[s]:
                    num_node += sum(hist.h)

                if (num_parent != 0) and (num_node != 0):
                    exploration = UCT_COEFF * (2 * log(num_parent) / num_node) ** 0.5

                    for hist in reward_per_scenario:
                        uct_value = hist.get_mean()

                        if uct_value > uct_max_on_actions:
                            uct_max_on_actions = uct_value

                    uct_per_scenario.append(uct_max_on_actions + exploration)
                    idx_scenarios.append(s)

            # mean on the scenarios which expanded this node
            mean = np.dot(uct_per_scenario, [self.probability[i] for i in idx_scenarios])

            return mean

    def integrate_buffer(self, Master_nodes):
        """
        Integrates the buffer of update from this scenario (the buffer is an attribute of the :class:`Tree`) \
        . The buffer is a list of updates coming from the worker. \
        One update is a list : [scenarioId, newNodeHash, parentHash, action, reward]

        :param dict Master_nodes: `Manager <https://docs.python.org/2/library/multiprocessing.html#sharing-state-\
        between-processes>`_ (Dictionary of :class:`master_node.MasterNode`) which saves the nodes of every scenario.
        """

        for update in self.buffer:

            scenarioId, newNodeHash, parentHash, action, reward = update

            node = Master_nodes.get(newNodeHash, 0)
            # new node if it doesn't exist
            if node == 0:
                node = MasterNode(self.numScenarios, nodehash=newNodeHash,
                                  parentNode=Master_nodes[parentHash], action=action)

            # add the reward in the expanded node
            node.add_reward(scenarioId, reward)

            # Back propagation
            Master_nodes[newNodeHash] = node
            while node.parentNode is not None:
                parent_hash = Master_nodes[node.hash].parentNode.hash
                parent_node = Master_nodes[parent_hash]
                parent_node.add_reward_action(scenarioId, node.arm, reward)
                Master_nodes[parent_hash] = parent_node
                node = parent_node

    @staticmethod
    def is_state_at_dest(destination, stateA, stateB):
        """
        Determines if the boat has gone beyond the destination. In this case, computes how much the boat \
        has overshot.

        :param destination: Destination state (goal).
        :param stateA: Previous state of the simulator.
        :param stateB: Current state of the simulator.

        :return: [True, frac] if the boat has reached the destination (frac is the last iteration proportion \
        corresponding to the part stateA--destination). Returns [False, None] otherwise.
        """
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
        """
        Determines if a state is considered as terminal (time is equal or larger than the time horizon, or if the boat is \
        out of the zone of interest).

        :param Simulator simulator: Simulator of the scenario.
        :param list state: State one wants to test [time, lat, lon]
        :return: True if the state is considered as terminal, False otherwise.
        """
        if simulator.times[state[0]] == simulator.times[-1]:
            return True

        elif state[1] < simulator.lats[0] or state[1] > simulator.lats[-1]:
            return True

        elif state[2] < simulator.lons[0] or state[2] > simulator.lons[-1]:
            return True
        else:
            return False

    def is_node_terminal(self, node):
        """
        Checks if the corresponding time of a node is equal to the time horizon of the simulation.

        :param worker.Node node: The node that is checked
        :return: True if the node is terminal, False otherwise.
        """
        return self.Simulator.times[node.depth] == self.TimeMax
