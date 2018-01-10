import master as mt

tree = mt.MasterTree.load_tree("tree_for_vis_200")

tree.plot_best_policy(grey=True)
for i in range(5):
    tree.plot_best_policy(grey=True, idscenario=i)
