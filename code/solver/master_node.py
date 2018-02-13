import sys

sys.path.append("../model/")
from utils import Hist
from simulatorTLKT import A_DICT, ACTIONS
import numpy as np
import matplotlib.pyplot as plt


class MasterNode:
    """
    Node of a MasterTree.

    :ivar int hash: hash of the node (key of the dictionary :py:attr:`MasterTree.nodes`)
    :ivar int arm: Action taken to get to this node from its parent.
    :ivar MasterNode parentNode: parent of this node
    :ivar numpy.array rewards: Array of `Hist`. Its shape is (#scenario, #possible actions).
    :ivar list children: List of children (:py:class:`MasterNode`)
    :ivar int depth: Depth of the node.
    """

    def __init__(self, numscenarios, nodehash=None, parentNode=None, action=None, rewards=[]):
        self.hash = nodehash
        self.arm = action
        self.parentNode = parentNode
        self.children = []
        self.depth = None
        self.guessed_rewards = dict()
        if len(rewards) == 0:
            self.rewards = np.array([[Hist() for _ in range(len(ACTIONS))] for _ in range(numscenarios)])
        else:
            self.rewards = np.array([[Hist(init.h) for init in rewards[ii]] for ii in range(len(rewards))])

    def add_reward(self, idscenario, reward):
        """
        Includes a reward into the histogram for a random action of one scenario.

        :param int idscenario: id of the scenario/workertree from which the update is coming.
        :param float reward: reward of the update.
        """
        # choose a random action
        action = np.random.randint(len(ACTIONS))
        # add the reward
        self.rewards[idscenario, action].add(reward)

    def add_reward_action(self, idscenario, action, reward):
        """
        Includes a reward into the histogram for one action of one scenario.

        :param int idscenario: id of the scenario/workertree from which the update is coming.
        :param int action: Action (in degree) of the update
        :param float reward: reward of the update
        """
        self.rewards[idscenario, A_DICT[action]].add(reward)

    def is_expanded(self, idscenario):
        """
        Check if this node has been expanded by a scenario.

        :param idscenario: id of the scenario
        :return boolean: True if the scenario has expanded this node.
        """
        return not all(hist.is_empty() for hist in self.rewards[idscenario, :])


def deepcopy_dict(nodes):
    """
    Return a deep copy of a MasterNode dictionary.
    Add also the children, the parentNode and the depth of each node.
    This method is called after the search before saving the result.

    :param dict nodes: a dictionary with MasterNode object
    :return: the deep copy of the input dictionary
    """
    new_dict = dict()
    n = len(nodes[hash(tuple([]))].rewards)  # compute the number of scenario

    for k, node in nodes.items():
        new_dict[k] = MasterNode(n, nodehash=node.hash, parentNode=None, action=node.arm,
                                 rewards=node.rewards)
        new_node = new_dict[k]
        if not node.parentNode:
            new_node.parentHash = None
        else:
            new_node.parentHash = node.parentNode.hash

        # Remove the node to optimize memory
        del nodes[k]

    # Add the parentNode and the children for each node of the dictionary
    for node in new_dict.values():
        if not node.parentHash:
            node.parentNode = None
        else:
            node.parentNode = new_dict[node.parentHash]
            node.parentNode.children.append(node)

    # Add the depth of each node as attribute
    node = new_dict[hash(tuple([]))]
    list_nodes = [node]
    node.depth = 0
    while list_nodes:
        node = list_nodes.pop(0)
        for n in node.children:
            list_nodes.append(n)
            n.depth = node.depth + 1

    return new_dict
