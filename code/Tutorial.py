""""""""" Tutorial Notebook for IBoat - PMCTS """""""""

""""""" Weather forecasts and Simulators """""""

""" Downloading weather forecasts """

import os

os.chdir(os.getcwd() + '/solver/')  # just to make sure you're working in the proper directory
import forest as ft
from master_node import deepcopy_dict
from master import MasterTree

# The starting day of the forecast. If it's too ancient, the forecast might not be available anymore
mydate = '20180228'  # for February 2, 2018

# We will download the mean scenario (id=0) and the first 2 perturbed scenarios
scenario_ids = range(3)
ft.download_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360], scenario_ids=scenario_ids)

""" Loading weather objects """

# We will download the mean scenario (id=0) and the first 2 perturbed scenarios
scenario_ids = range(3)
Weathers = ft.load_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360], scenario_ids=scenario_ids)

""" Create simulators and displaying wind conditions """

NUMBER_OF_SIM = 3  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 44, 355]
N_DAYS_SIM = 3  # time horizon in days

Sims = ft.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                            stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)
ft.play_multiple_scenarios(Sims)

""""""" The Parallel Monte-Carlo Tree Search """""""

""" Initialisation of the search """

missionheading = 0  # direction wrt. true North we want to go the furthest.
ntra = 50  # number of trajectories used during the initialization
destination, timemin = ft.initialize_simulators(Sims, ntra, STATE_INIT, missionheading, plot=True)
print("destination : {} & timemin : {}".format(destination, timemin))

""" Create a Forest and launch a PMCTS """

budget = 100  # number of nodes we want to expand in each worker
frequency = 10  # number of steps performed by worker before writing the results into the master
forest = ft.Forest(listsimulators=Sims, destination=destination, timemin=timemin, budget=budget)
master_nodes = forest.launch_search(STATE_INIT, frequency)
new_dict = deepcopy_dict(master_nodes)
forest.master = MasterTree(Sims, destination, nodes=new_dict)
forest.master.get_best_policy()
forest.master.plot_tree_uct()
forest.master.plot_tree_uct(1)
forest.master.plot_hist_best_policy(interactive=True)
forest.master.save_tree("my_tuto_results")
