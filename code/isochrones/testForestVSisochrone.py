import sys
sys.path.append("../model")
sys.path.append("../solver")
sys.path.append("../isochrones")
import forest as ft
from simulatorTLKT import Simulator, HOURS_TO_DAY
import numpy as np
from worker import Tree
import IsochroneClass as IC
from weatherTLKT import Weather
import simulatorTLKT as SimC

# parameters
name = "tree_for_vis_100_20"
frequency = 10
budget = 10000

mydate = '20180130'

#ft.download_scenarios(mydate,latBound = [50-28, 50],lonBound = [-40 + 360, 360])
#
Weathers = ft.Forest.load_scenarios(mydate, latBound=[40, 50], lonBound=[360 - 15, 360])

# We create N simulators based on the scenarios
NUMBER_OF_SIM = 20  # <=20
SIM_TIME_STEP = 6  # in hours
STATE_INIT = [0, 44, -5 + 360]
N_DAYS_SIM = 8  # time horizon in days

sims = ft.Forest.create_simulators(Weathers, numberofsim=NUMBER_OF_SIM, simtimestep=SIM_TIME_STEP,
                                   stateinit=STATE_INIT, ndaysim=N_DAYS_SIM)

# initialize the simulators to get common destination and individual time min

missionheading = 0
ntra = 50

destination, timemin = ft.Forest.initialize_simulators(sims, ntra, STATE_INIT, missionheading, plot=True)

print("destination : " + str(destination) + "  &  timemin : " + str(timemin) + "\n")

# départ = [44, 355]
# arrivée = [46.92261598492173, 355.0]

#%%
#forest = ft.Forest(listsimulators=sims, destination=destination, timemin=timemin, budget=budget)
#forest.launch_search(STATE_INIT, frequency)
#forest.master.save_tree(name)

#%%

mydate = '20180130'
modelcycle = '00z'
pathToSaveObj = '../data/' + mydate + '_gfs_1p00_' + modelcycle + '.obj'
Wavg = Weather.load(pathToSaveObj)

sim = ft.Forest.create_simulators([Wavg], 1, stateinit=STATE_INIT)[0]
SimC.Boat.UNCERTAINTY_COEFF = 0

sim.reset(STATE_INIT)
solver_iso = IC.Isochrone(sim, STATE_INIT[1:3], destination, delta_cap=5, increment_cap=18, nb_secteur=200,
                          resolution=200)
temps, politique, politique_finale, trajectoire = solver_iso.isochrone_methode()

m = sim.prepareBaseMap(centerOfMap=STATE_INIT[1:], proj='aeqd')
for i, el in enumerate(trajectoire):
    trajectoire[i] = [i] + el

sim.plotTraj(trajectoire, m, quiv=True, scatter=False)

#%%
SimC.Boat.UNCERTAINTY_COEFF = 0.2

nb_actions = 14 #amélioration notable entre 12 et 14 actions appliquées
politique_testee = politique[0:nb_actions]

somme_temps = 0
temps_tous_scenario = []
moyenne_temps_1_scenario = 0

variance_tous_scenario = []
variance_1_scenario = 0
stock = []
stock_tout = []


for i in range(len(sims)):
    moyenne_temps_1_scenario = 0
    variance_1_scenario = 0
    stock = []
    for j in range(30):
        sims[i].reset(STATE_INIT)
        for action in politique_testee:
            sims[i].doStep(action)
        atDest = False
        terminal = False
        frac = 0
        while (not atDest) and (not terminal):
            Ddest,cap_a_suivre = sims[i].getDistAndBearing(sims[i].state[1:3],destination)
            sims[i].doStep(cap_a_suivre)
            atDest,frac =Tree.is_state_at_dest(destination,sims[i].prevState,sims[i].state)
            terminal = Tree.is_state_terminal(sims[i],sims[i].state)
        if atDest:
            temps_scenario = sims[i].times[sims[i].state[0]]-(1-frac)*(SIM_TIME_STEP/24)
        else:
            temps_scenario = sims[i].times[-1]
        moyenne_temps_1_scenario += temps_scenario
        stock.append(temps_scenario)
        stock_tout.append(temps_scenario)
    moyenne_temps_1_scenario = moyenne_temps_1_scenario/30
    temps_tous_scenario.append(moyenne_temps_1_scenario)
    somme_temps += moyenne_temps_1_scenario
    for value in stock:
        variance_1_scenario += (moyenne_temps_1_scenario - value)**2
    variance_1_scenario = variance_1_scenario/30
    variance_tous_scenario.append(variance_1_scenario)

for i in range(len(temps_tous_scenario)):
    print("temps scénario ",i+1," = ", temps_tous_scenario[i])
    print("variance scénario   = ", variance_tous_scenario[i])
    print()

moyenne_iso = somme_temps/len(sims)
print("moyenne des temps isochrones                   = ", moyenne_iso)

Var_tot = 0
for value in stock_tout:
    Var_tot += (moyenne_iso - value)**2
Var_tot = Var_tot/len(stock_tout)
print("variance globale des isochrones                = ", Var_tot)

Var = 0
for value in temps_tous_scenario:
    Var += (moyenne_iso - value)**2
Var = Var/len(temps_tous_scenario)
print("variance des moyennes des scénarios isochrones = ", Var)

print()
print("temps scénario vent moyen isochrones = ", temps)

# Idem pour le MCTS qu'avec les isochrones. On change juste :          
#politique_testee = politique_MCTS[nb_actions]
#
#
# Comparaison des temps moyens et variances obtenus par scénario et au global

print('test')

IC.comparateur_plans_optimaux(sims,30,STATE_INIT,destination,politique,np.zeros(20),True)
