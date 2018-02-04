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

    def backup(self, idscenario, reward):
        """
        Propagates the reward through the master tree, starting from this node.

        :param int idscenario: id of the scenario/workertree where the update is coming
        :param float reward: reward of the update
        """
        parent = self.parentNode
        if parent is not None:
            parent.add_reward_action(idscenario, self.arm, reward)
            parent.backup(idscenario, reward)

    def is_expanded(self, idscenario):
        """
        Check if this node has been expanded by a scenario.

        :param idscenario: id of the scenario
        :return boolean: True if the scenario has expanded this node.
        """
        return not all(hist.is_empty() for hist in self.rewards[idscenario, :])

    # TODO enelever les deux fonction suivantes si on ne s'en sert pas
    def plot_hist(self, idscenario, action):
        # print(self.rewards[idscenario, action].h)
        fig, ax = plt.subplots()
        plt.bar(x=Hist.MEANS, height=self.rewards[idscenario, action].h, width=Hist.THRESH[1] - Hist.THRESH[0])
        fig.show()
        return fig

    def plot_mean_hist(self, action, probability):
        # Mean on all the scenarios:
        hist = sum(self.rewards[ii, action].h * probability[ii] for ii in range(len(self.rewards[:, action])))

        fig, ax = plt.subplots()
        # print(hist)
        plt.bar(x=Hist.MEANS, height=hist, width=Hist.THRESH[1] - Hist.THRESH[0])
        fig.show()
        return fig


def deepcopy_dict(nodes):
    """
    Fais la copie et le get children en meme temps !
    :param nodes:
    :return:
    """
    new_dict = dict()
    n = len(nodes[hash(tuple([]))].rewards)

    for k, node in nodes.items():
        new_dict[k] = MasterNode(n, nodehash=node.hash, parentNode=None, action=node.arm,
                                 rewards=node.rewards)
        new_node = new_dict[k]
        if not node.parentNode:
            new_node.parentHash = None
        else:
            new_node.parentHash = node.parentNode.hash

    for node in new_dict.values():
        if not node.parentHash:
            node.parentNode = None
        else:
            node.parentNode = new_dict[node.parentHash]
            node.parentNode.children.append(node)

    return new_dict
