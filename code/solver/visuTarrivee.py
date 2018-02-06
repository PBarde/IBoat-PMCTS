import forest as ft
from worker import Tree
from math import exp
import matplotlib.pyplot as plt

mydate = '20180108'

# ft.download_scenarios(mydate, latBound=[40, 50], lonBound=[-15 + 360, 360])
#
Weathers = ft.load_scenarios(mydate, latBound=[40, 50], lonBound=[360 - 15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 1  # <=20
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

destination, timemin = ft.initialize_simulators(sims, ntra, STATE_INIT, missionheading)
print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")
forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin)

ntra = 100
tree = forest.workers[0]
endpoint = []
endtimes = []
rewards = []


for i in range(ntra):

    tree.Simulator.state = list(STATE_INIT)

    dist, action = tree.Simulator.getDistAndBearing(tree.Simulator.state[1:], tree.destination)
    atDest, frac = Tree.is_state_at_dest(tree.destination, tree.Simulator.prevState, tree.Simulator.state)

    while (not atDest) \
            and (not Tree.is_state_terminal(tree.Simulator, tree.Simulator.state)):
        tree.Simulator.doStep(action)
        dist, action = tree.Simulator.getDistAndBearing(tree.Simulator.state[1:], tree.destination)
        atDest, frac = Tree.is_state_at_dest(tree.destination, tree.Simulator.prevState, tree.Simulator.state)

    if atDest:
        finalTime = tree.Simulator.times[tree.Simulator.state[0]] - (1 - frac) * (
            tree.Simulator.times[tree.Simulator.state[0]] - tree.Simulator.times[tree.Simulator.state[0] - 1])
        reward = (exp((tree.TimeMax * 1.001 - finalTime) / (tree.TimeMax * 1.001 - tree.TimeMin)) - 1) / (
            exp(1) - 1)
        endpoint.append(list(tree.Simulator.state))
        endtimes.append(finalTime)
        rewards.append(reward)
    else:
        reward = 0


basemap = tree.Simulator.prepareBaseMap()
tree.Simulator.plotTraj(endpoint, basemap, scatter=True)

xd,yd = basemap(destination[1], destination[0])
basemap.scatter(xd, yd, color="red")


xs,ys = basemap(STATE_INIT[2], STATE_INIT[1])
basemap.scatter(xs, ys, color="g")

for ii, endstate in enumerate(endpoint):
    x, y = basemap(endstate[2], endstate[1])
    plt.annotate("T = "+"%.2f"%endtimes[ii] + " reward = "+"%.2f"%rewards[ii], (x, y))


plt.show()



