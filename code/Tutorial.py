""""""""" Tutorial Notebook for IBoat - PMCTS """""""""

""""""" Weather forecasts and Simulators """""""

""" Downloading weather forecasts """

import os

os.chdir(os.getcwd() + '/solver/')  # just to make sure you're working in the proper directory
import forest as ft
from master_node import deepcopy_dict
from master import MasterTree
import time
import worker
import sys
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../isochrones/')
sys.path.append('../model/')
import isochrones as IC
from simulatorTLKT import Boat

# The starting day of the forecast. If it's too ancient, the forecast might not be available anymore
mydate = time.strftime("%Y%m%d")  # mydate = '20180228'# for February 2, 2018

# We will download the mean scenario (id=0) and the first 2 perturbed scenarios
scenario_ids = range(3)
ft.download_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360], scenario_ids=scenario_ids)

""" Loading weather objects """

# We will download the mean scenario (id=0) and the first 2 perturbed scenarios
Weathers = ft.load_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360], scenario_ids=scenario_ids)

""" Create simulators and displaying wind conditions """

Boat.UNCERTAINTY_COEFF = 0.2  # Characterizes the uncertainty on the boat's dynamics
NUMBER_OF_SIM = 3  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 44., 355.]
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

##Exploration Parameters##
worker.RHO = 0.5  # Proportion between master utility and worker utility of node utility.
worker.UCT_COEFF = 1 / 2 ** 0.5  # Exploration coefficient in the UCT formula.


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

""""""" Isochrones """""""

""" Launch a search """

Boat.UNCERTAINTY_COEFF = 0
NUMBER_OF_SIM = 3  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 44., 355.]
N_DAYS_SIM = 3

sim = ft.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                            stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)[0]

solver_iso = IC.Isochrone(sim, STATE_INIT, destination, delta_cap=5, increment_cap=9, nb_secteur=300,
                          resolution=100)
temps_estime, plan_iso, plan_iso_ligne_droite, trajectoire = solver_iso.isochrone_methode()

print(temps_estime)
IC.plot_trajectory(sim, trajectoire, quiv=True)

""" Comparision """
plan_PMCTS = forest.master.best_policy[-1]

Boat.UNCERTAINTY_COEFF = 0.2

mean_PMCTS, var_PMCTS = IC.estimate_perfomance_plan(Sims, ntra, STATE_INIT, destination, plan_PMCTS, plot=True, verbose=False)

mean_iso, var_iso = IC.estimate_perfomance_plan(Sims, ntra, STATE_INIT, destination, plan_iso, plot=True, verbose=False)

IC.plot_comparision(mean_PMCTS, var_PMCTS, mean_iso, var_iso, ["PCMTS", "Isochrones"])
