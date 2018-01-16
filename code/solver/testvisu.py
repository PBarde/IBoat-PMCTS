import master as mt
import numpy as np

id = None
tree = mt.MasterTree.load_tree("tree_for_vis_1000")
tree.get_best_policy()
# tree.plot_best_policy(grey=True, idscenario=id)

tree.plot_hist_best_policy(idscenario=id)
#
# for i in range(tree.numScenarios):
#     tree.plot_best_policy(grey=True, idscenario=i)

# node0 = tree.nodes[hash(tuple([]))]
# node0.plot_hist(2, 2)
# node0.plot_mean_hist(2, tree.probability)
