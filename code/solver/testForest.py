import forest as ft
from simulatorTLKT import Simulator, HOURS_TO_DAY
import numpy as np
from worker import Tree

# parameters
name = "tree_for_vis_100_20_bis"
frequency = 10
budget = 20

mydate = '20180108'

# ft.Forest.download_scenarios(mydate,latBound = [50-28, 50],lonBound = [-40 + 360, 360])
#
Weathers = ft.Forest.load_scenarios(mydate, latBound=[40, 50], lonBound=[360 - 15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 20  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 47.5, -3.5 + 360]
N_DAYS_SIM = 8  # time horizon in days

sims = ft.Forest.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                                   stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)

# initialize the simulators to get common destination and individual time min

missionheading = 235
ntra = 50

destination, timemin = ft.Forest.initialize_simulators(sims, ntra, STATE_INIT, missionheading)
print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")

# destination= [44.818714942905117, 350.99630857217124]
# timemin= 5.2017906821

forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin, budget=budget)
forest.launch_search(STATE_INIT, frequency)
forest.master.get_children()
forest.master.get_depth()
forest.master.get_best_policy()
forest.master.save_tree(name)

