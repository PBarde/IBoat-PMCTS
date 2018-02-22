#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import log, sin, cos, pi
from matplotlib import animation
import pickle
from utils import Hist
from worker import UCT_COEFF
from master_node import MasterNode

sys.path.append('../model/')
from simulatorTLKT import ACTIONS, Simulator, A_DICT


class MasterTree:
    """
    Tree that represent the final result of a MCTS parallel search on multiple scenarios.

    :ivar dict nodes: dictionary containing `MasterNode`, the keys are their corresponding hash
    :ivar numpy.array probability: array containing the probability of each scenario
    :ivar list Simulators: List of the simulators used during the search
    :ivar int numScenarios: Number of scenarios
    :ivar list destination: Destination state [lat, long]
    :ivar dict best_policy: Dictionary of list of actions. Key of the dictionary is the scenario id.\
    Key -1 is for the average policy. The list is the sequel of action.
    :ivar dict best_global_nodes_policy: Dictionary of list of MasterNodes encountered during the best\
     policy of one scenario. The Key of the dictionary is the scenario id. Key -1 is for the average policy.
    """

    def __init__(self, sims, destination, nodes=dict(), proba=[]):
        num_scenarios = len(sims)
        self.nodes = nodes
        if len(nodes) == 0:
            self.nodes[hash(tuple([]))] = MasterNode(num_scenarios, nodehash=hash(tuple([])))

        self.Simulators = sims
        self.numScenarios = num_scenarios
        self.destination = destination
        self.best_policy = dict()
        self.best_nodes_policy = dict()
        if len(proba) != num_scenarios:
            self.probability = np.array([1 / num_scenarios for _ in range(num_scenarios)])
        else:
            self.probability = np.array(proba)

    def get_best_policy(self):
        """
        Compute the best policy for each scenario and the global best policy.
        """

        # Get best policy for each scenario and the global one:
        for id_scenario in range(-1, len(self.Simulators)):
            if id_scenario == -1:
                print("Global policy")
            else:
                print("Policy for scenario {}".format(id_scenario))
            nodes_policy = [self.nodes[hash(tuple([]))]]  # rootNode
            policy = []
            node = nodes_policy[0]
            while node.children:
                child, action = self.get_best_child(node, idscenario=id_scenario, verbose=True)

                # If in this scenario the node has zero child, stop the loop
                if child is None:
                    break

                # Save the best node and the best action
                nodes_policy.append(child)
                policy.append(action)
                node = child

            self.best_policy[id_scenario] = policy
            self.best_nodes_policy[id_scenario] = nodes_policy

    def get_best_child(self, node, idscenario=-1, verbose=False):
        """
        Compare the children of a node based on their reward and return the best one.

        :param MasterNode node: the parent node
        :param int idscenario: id of the considered scenario. If default (None), the method return the best child\
         for the global tree
        :param bool verbose: If True, print the best reward and the best action.
        :return: A tuple: (the best child, the action taken to go from the node to its best child)
        """
        best_reward = 0
        best_action = None
        best_child = None

        # Test if at least one node has been expanded by the scenario
        if idscenario is not -1:
            if all(not child.is_expanded(idscenario) for child in node.children):
                return best_child, best_action

        for child in node.children:
            value, _ = self.get_utility(child, idscenario)
            if value > best_reward:
                best_reward = value
                best_child = child
                best_action = child.arm
        if verbose:
            print("Depth {}: best reward = {} for action = {}".format(best_child.depth, best_reward, best_action))
        return best_child, best_action

    def guess_reward(self, node, idscenario):
        father = node.parentNode
        if father.is_expanded(idscenario):
            _, exploration = self.get_utility(father, idscenario)
            grandfather = father.parentNode
            best_child_grandfather, _ = self.get_best_child(grandfather, idscenario)
            value_grd, expl_grd = self.get_utility(best_child_grandfather, idscenario)
            guessed_reward = value_grd + expl_grd - exploration
        else:
            guessed_reward = father.guessed_rewards[idscenario]

        return guessed_reward

    def get_utility(self, node, idscenario):

        # Test if the node has been expanded by the scenario
        if idscenario is not -1:
            if not node.is_expanded(idscenario):
                return 0, 0

        num_parent = 0
        num_node = 0
        reward_per_action = np.zeros(shape=len(ACTIONS))
        for j in range(len(ACTIONS)):
            if idscenario is -1:
                reward_per_action_per_scenario = []
                for i in range(self.numScenarios):
                    if node.is_expanded(i):
                        reward_per_action_per_scenario.append(node.rewards[i, j].get_mean())
                        num_node += sum(node.rewards[i, j].h)
                        num_parent += sum(node.parentNode.rewards[i, j].h)
                    else:
                        reward_guess = self.guess_reward(node, i)
                        node.guessed_rewards[i] = reward_guess
                        reward_per_action_per_scenario.append(reward_guess)

                reward_per_action[j] = np.dot(reward_per_action_per_scenario,
                                              self.probability)
            else:
                num_node += sum(node.rewards[idscenario, j].h)
                num_parent += sum(node.parentNode.rewards[idscenario, j].h)
                reward_per_action[j] = node.rewards[idscenario, j].get_mean()

        value = np.max(reward_per_action)
        exploration = UCT_COEFF * (2 * log(num_parent) / num_node) ** 0.5

        return value, exploration

    def get_points(self, node, points, probability, coordinate=(0, 0), idscenario=-1, objective="depth"):
        """
        Recursive function used in :py:meth:`plot_tree` and :py:meth:`plot_tree_colored` to compute the coordinates\
        of a node in the plot, and another value

        :param node: a MasterNode object
        :param list points: the previous list of points
        :param np.array probability: probability of each scenario
        :param tuple coordinate: coordinates of the previous point
        :param int idscenario: id of the corresponding worker tree to be plot. If None (default), the global tree is plotted.
        :return: the expanded list of points
        """
        x0, y0 = coordinate
        for child in node.children:
            if idscenario is not -1:
                if not child.is_expanded(idscenario):
                    continue
            x = x0 + sin(child.arm * pi / 180)
            y = y0 + cos(child.arm * pi / 180)

            if child.parentNode is not None:
                if objective == "uct":
                    value, exploration = self.get_utility(child, idscenario)
                    points.append((x0, y0, x, y, value + exploration, value, exploration))
                elif objective == "depth":
                    points.append((x0, y0, x, y, child.depth))

            points = self.get_points(child, points, probability, coordinate=(x, y), idscenario=idscenario,
                                     objective=objective)
        return points

    def plot_tree(self, idscenario=-1, number_subplots=1):
        """
        Plot a 2D representation of a tree.

        :param boolean grey: if True, each node/branch are plot with a color (grey scale) depending of the depth of the node
        :param int idscenario: id of the corresponding worker tree to be plot. If -1 (default), the global tree is plotted.
        :return: A tuple (fig, ax) of the current plot
        """
        node = self.nodes[hash(tuple([]))]  # rootNode
        # Get the coordinates and the depth
        points = self.get_points(node, [], self.probability, idscenario=idscenario, objective="depth")
        coordinates = [[i[0] for i in points], [i[1] for i in points], [i[2] for i in points], [i[3] for i in points]]
        depth = [i[4] for i in points]

        # Plots
        fig = plt.figure()
        ax = fig.add_subplot(1, number_subplots, 1)
        self.draw_points(ax, coordinates, depth)
        ax.set_title("Depth")
        fig.show()
        return fig, ax

    def draw_points(self, ax, coordinates, values):
        for i in range(len(values)):
            ax.plot([coordinates[0][i], coordinates[2][i]], [coordinates[1][i], coordinates[3][i]], color="grey",
                    linewidth=0.5, zorder=1)
        sc = ax.scatter(coordinates[2], coordinates[3], c=values,
                        s=np.dot(np.power(values, 2), 32 / np.power(max(values), 2)), zorder=2,
                        cmap="Reds")
        plt.colorbar(sc)
        ax.plot(0, 0, color="blue", marker='o', markersize='10')
        plt.axis('equal')

    def plot_tree_uct(self, idscenario=-1):
        """
        Plot a the tree 3 times: first one the colormap represents the sum of exploitation and exploration for each node
        , the second one represents the exploitation and the third one the exploration.

        :param int idscenario: id of the corresponding worker tree to be plot. If -1 (default), the global tree is plotted.
        :return: A tuple (fig, ax) of the current plot
        """

        node = self.nodes[hash(tuple([]))]  # rootNode

        # Get the coordinates and the values
        points = self.get_points(node, [], self.probability, idscenario=idscenario, objective="uct")
        coordinates = [[i[0] for i in points], [i[1] for i in points], [i[2] for i in points], [i[3] for i in points]]
        total = [i[4] for i in points]
        exploitation = [i[5] for i in points]
        exploration = [i[6] for i in points]

        # Plots
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        self.draw_points(ax, coordinates, total)
        ax.set_title("Total utility")

        ax = fig.add_subplot(1, 3, 2)
        self.draw_points(ax, coordinates, exploitation)
        ax.set_title("Exploitation")

        ax = fig.add_subplot(1, 3, 3)
        self.draw_points(ax, coordinates, exploration)
        ax.set_title("Exploration")

        fig.show()
        return fig, ax

    def plot_best_policy(self, idscenario=-1, number_subplots=1):
        """
        Plot a representation of a tree and its best policy.

        :param boolean grey: if True, each node/branch are plot with a color (grey scale) depending of the depth of the node
        :param int idscenario: id of the corresponding worker tree to be plot. If -1 (default), the global tree is plotted.
        :return: A tuple (fig, ax) of the current plot
        """
        # check if the best_policy has been computed
        if len(self.best_policy) == 0:
            self.get_best_policy()

        # Get the right policy
        nodes_policy = self.best_nodes_policy[idscenario]

        fig, ax = self.plot_tree(idscenario=idscenario, number_subplots=number_subplots)
        x0 = 0
        y0 = 0
        length = 1
        for node in nodes_policy[1:]:
            x = x0 + length * sin(node.arm * pi / 180)
            y = y0 + length * cos(node.arm * pi / 180)
            ax.plot([x0, x], [y0, y], color="red", marker='o', markersize='6')
            x0 = x
            y0 = y
        return fig, ax

    def plot_hist_best_policy(self, idscenario=-1):
        """
        Plot the best policy as in :py:meth:`plot_best_policy`, with the histogram of the best action at each node\
         (`Animation <https://matplotlib.org/api/animation_api.html>`_)

        :param int idscenario: id of the corresponding worker tree to be plot. If -1 (default), the global tree is plotted.
        :return: the `figure <https://matplotlib.org/api/figure_api.html>`_ of the current plot
        """
        # check if the best_policy has been computed
        if len(self.best_policy) == 0:
            self.get_best_policy()

        # Get the right policy:
        nodes_policy = self.best_nodes_policy[idscenario]
        policy = self.best_policy[idscenario]

        # Plot
        fig, ax1 = self.plot_best_policy(idscenario=idscenario, number_subplots=2)

        ax2 = fig.add_subplot(1, 2, 2)
        barcollection = ax2.bar(x=Hist.MEANS, height=[0 for _ in Hist.MEANS],
                                width=Hist.THRESH[1] - Hist.THRESH[0])
        pt, = ax1.plot(0, 0, color="green", marker='o', markersize='7')
        x0, y0 = 0, 0
        x_list = [x0]
        y_list = [y0]
        for node in nodes_policy[1:]:
            x = x0 + 1 * sin(node.arm * pi / 180)
            y = y0 + 1 * cos(node.arm * pi / 180)
            x_list.append(x)
            y_list.append(y)
            x0, y0 = x, y

        def animate(i):
            n = nodes_policy[i]
            if i == len(nodes_policy) - 1:
                # last nodes: same reward for all actions
                a = 0
            else:
                a = A_DICT[policy[i]]
            if idscenario is -1:
                hist = sum(n.rewards[ii, a].h * self.probability[ii] for ii in range(len(n.rewards[:, a])))
            else:
                hist = n.rewards[idscenario, a].h
            for j, b in enumerate(barcollection):
                b.set_height(hist[j])
            ax2.set_ylim([0, np.max(hist) + 1])
            pt.set_data(x_list[i], y_list[i])

            return barcollection, pt

        anim = animation.FuncAnimation(fig, animate, frames=len(nodes_policy), interval=1000, blit=False)
        plt.show()
        return fig

    def save_tree(self, name):
        """
        Save the master tree (object) in the data Folder.

        :param name: Name of the file.
        """
        filehandler = open("../data/" + name + '.pickle', 'wb')
        pickle.dump(self, filehandler)
        filehandler.close()

    @classmethod
    def load_tree(cls, name):
        """
        Load a master tree (object) from the data Folder.

        :param name: Name of the file.
        """
        filehandler = open("../data/" + name + '.pickle', 'rb')
        loaded_tree = pickle.load(filehandler)
        filehandler.close()
        return loaded_tree
