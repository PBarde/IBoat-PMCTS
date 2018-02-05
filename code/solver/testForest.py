import forest as ft
from master_node import deepcopy_dict
from master import MasterTree

# parameters
name = "tree_0130_test"
frequency = 100
budget = 1000

mydate = '20180130'

# ft.download_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360])
#
Weathers = ft.load_scenarios(mydate, latBound=[40, 50], lonBound=[360 - 15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 4  # <=20
SIM_TIME_STEP = 6  # in hours
# STATE_INIT = [0, 47.5, -3.5 + 360]
STATE_INIT = [0, 42.5, -10 + 360]

N_DAYS_SIM = 4  # time horizon in days

sims = ft.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                            stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)

# ft.play_multiple_scenarios(sims)
# sims[0].play_scenario()
# initialize the simulators to get common destination and individual time min

missionheading = 0
ntra = 50

destination, timemin = ft.initialize_simulators(sims, ntra, STATE_INIT, missionheading, plot=True)
print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")
# destination = [44.90198407864892, 350.5981540154125]
# timemin = 5.75739048005

forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin, budget=budget)
master_nodes = forest.launch_search(STATE_INIT, frequency)
new_dict = deepcopy_dict(master_nodes)
forest.master = MasterTree(sims, destination, nodes=new_dict)
forest.master.save_tree(name)
