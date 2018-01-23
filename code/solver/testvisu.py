import master as mt
import sys
sys.path.append('../model/')
from simulatorTLKT import ACTIONS, Simulator, A_DICT
import numpy as np
if len(sys.argv) > 1:
    id = int(sys.argv[1])
else:
    id = None

# tree = mt.MasterTree.load_tree("tree_for_vis_1000")
tree = mt.MasterTree.load_tree("tree_for_vis_20_20_p05")

tree.get_best_policy()
# tree.plot_best_policy(grey=True, idscenario=id)
tree.plot_tree(idscenario=id)
tree.plot_hist_best_policy(idscenario=id)
#
# for i in range(tree.numScenarios):
#     tree.plot_best_policy(grey=True, idscenario=i)

node0 = tree.nodes[hash(tuple([]))]
pol = tree.best_policy[17] + [0]
for i, node in enumerate(tree.best_nodes_policy[17]):
    pass
    # node.plot_hist(17, A_DICT[pol[i]])
print(node.is_expanded(17))
# node0.plot_hist(2, 2)
# node0.plot_mean_hist(action = 2, tree.probability)
# for action in range(8):
#     node0.plot_hist(idscenario=2, action=action)
