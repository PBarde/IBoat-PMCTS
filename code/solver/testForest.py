import forest as ft
from master import MasterTree
from master_node import deepcopy_dict

# parameters
name = "tree_test4CPU"
frequency = 10
budget = 100

mydate = '20180108'

# ft.Forest.download_scenarios(mydate,latBound = [50-28, 50],lonBound = [-40 + 360, 360])
#
Weathers = ft.Forest.load_scenarios(mydate, latBound=[40, 50], lonBound=[360 - 15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 4  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 47.5, -3.5 + 360]
N_DAYS_SIM = 8  # time horizon in days

sims = ft.Forest.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                                   stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)

# initialize the simulators to get common destination and individual time min

missionheading = 235
ntra = 50

# destination, timemin = ft.Forest.initialize_simulators(sims, ntra, STATE_INIT, missionheading)
# print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")
destination = [44.62224559323147, 350.9976771826662]
timemin = 5.2654198058866042

forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin, budget=budget)
master_nodes = forest.launch_search(STATE_INIT, frequency)

new_dict = deepcopy_dict(master_nodes)

forest.master = MasterTree(sims, destination, nodes=new_dict)
forest.master.get_depth()
forest.master.get_best_policy()
# forest.master.save_tree(name)
