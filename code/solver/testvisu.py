import master as mt
import numpy as np

tree = mt.MasterTree.load_tree("tree_for_vis_1000")

tree.plot_best_policy(grey=True)

for i in range(tree.numScenarios):
    tree.plot_best_policy(grey=True, idscenario=i)
