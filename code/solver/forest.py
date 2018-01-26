import worker as mt
import sys

sys.path.append('../model/')
from weatherTLKT import Weather
from simulatorTLKT import Simulator, HOURS_TO_DAY
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import Process

class Forest:
    """
    Object coordinating a MasterTree and its WorkerTrees
    """

    def __init__(self, listsimulators=[], destination=[], timemin=0, budget=100):
        # change the constant in master module
        self.workers = dict()
        nscenario = len(listsimulators)
        for i, sim in enumerate(listsimulators):
            self.workers[i] = mt.Tree(workerid=i, nscenario = nscenario, ite=0, budget=budget,
                                      simulator=deepcopy(sim), destination=deepcopy(destination), TimeMin=timemin)

    def launch_search(self, root_state, frequency, master):
        # Create the workers Process
        worker_process = dict()
        for worker in self.workers.values():
            worker_process[worker.id] = Process(name='worker' + str(worker.id), target=worker.uct_search,
                                                 args=(deepcopy(root_state), frequency, master))
        # Launch the threads
        for w_p in worker_process.values():
            w_p.start()


        # Wait for process to complete
        for w_th in worker_process.values():
            w_th.join()


        for worker in self.workers.values():
            print("Number of iterations for worker " + str(worker.id) + ": " + str(worker.ite))

    @staticmethod
    def download_scenarios(mydate, latBound=[43, 50], lonBound=[-10 + 360, 360],
                           website='http://nomads.ncep.noaa.gov:9090/dods/',
                           modelcycle=range(1, 21)):
        """
        To download the scenarios (launch in real python console)
        :param mydate:
        :param latBound:
        :param lonBound:
        :param website:
        :param modelcycle:
        :return:
        """
        pathToSaveObj = []
        for ii in modelcycle:

            if ii < 10:
                cycle = '0' + str(ii)
            else:
                cycle = str(ii)

            url = (website + 'gens/gens' + mydate + '/gep' + cycle + '_00z')
            pathToSaveObj.append(('../data/' + mydate + '_gep_' + cycle + '00z.obj'))

            Weather.download(url, pathToSaveObj[ii - 1], latBound=latBound, lonBound=lonBound, timeSteps=[0, 64],
                             ens=True)

    @staticmethod
    def load_scenarios(mydate, website='http://nomads.ncep.noaa.gov:9090/dods/',
                       modelcycle=range(1, 21), latBound=[-90, 90],
                       lonBound=[0, 360], timeSteps=[0, 64]):

        pathToSaveObj = []
        weather_scen = []
        for ii in modelcycle:

            if ii < 10:
                cycle = '0' + str(ii)
            else:
                cycle = str(ii)

            url = (website + 'gens/gens' + mydate + '/gep' + cycle + '_00z')
            pathToSaveObj.append(('../data/' + mydate + '_gep_' + cycle + '00z.obj'))

            weather_scen.append(Weather.load(pathToSaveObj[ii - 1], latBound, lonBound, timeSteps))

        return weather_scen

    @staticmethod
    def create_simulators(weathers, numberofsim, simtimestep=6, stateinit=[0, 47.5, -3.5 + 360],
                          ndaysim=8, delatlatlon=0.5):
        """

        :param weathers:
        :param numberofsim:
        :param simtimestep:
        :param stateinit:
        :param ndaysim:
        :param delatlatlon:
        :return:
        """
        sims = []
        for jj in range(numberofsim):
            # We shift the times so that all times are in the correct bounds for interpolations
            weathers[jj].time = weathers[jj].time - weathers[jj].time[0]

            # We set up the parameters of the simulation
            timemax = simtimestep * len(weathers[jj].time)
            times = np.arange(0, ndaysim, simtimestep * HOURS_TO_DAY)
            lats = np.arange(weathers[jj].lat[0], weathers[jj].lat[-1], delatlatlon)
            lons = np.arange(weathers[jj].lon[0], weathers[jj].lon[-1], delatlatlon)
            sims.append(Simulator(times, lats, lons, weathers[jj], stateinit))

        return sims

    @staticmethod
    def initialize_simulators(sims, ntra, stateinit, missionheading, plot=False):

        meanarrivaldistances = []
        ii = 0

        if plot:
            meantrajs_dest = []
            trajsofsim = []
            traj = []

        for sim in sims:
            arrivaldistances = []

            for _ in range(ntra):
                sim.reset(stateinit)

                if plot:
                    traj.append(list(sim.state))

                for t in sim.times[0:-1]:
                    sim.doStep(missionheading)

                    if plot:
                        traj.append(list(sim.state))

                if plot:
                    trajsofsim.append(list(traj))
                    traj = []

                dist, dump = sim.getDistAndBearing(stateinit[1:],(sim.state[1:]))
                arrivaldistances.append(dist)
                ii += 1

            meanarrivaldistances.append(np.mean(arrivaldistances))
            if plot:
                meantrajs_dest.append(np.mean(trajsofsim, 0))
                trajsofsim = []

        mindist = np.min(meanarrivaldistances)
        destination = sim.getDestination(mindist,missionheading,stateinit[1:])

        if plot:
            minarrivaltimes = []
            shape = (len(sims), len(sims[0].times), 3)
            meantrajs = []
            trajsofsim = np.full((ntra, len(sims[0].times), 3), stateinit)

        arrivaltimes = []

        for ii, sim in enumerate(sims):

            for jj in range(ntra):
                sim.reset(stateinit)

                if plot:
                    traj = []
                    traj.append(list(sim.state))

                dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                sim.doStep(action)

                if plot:
                    traj.append(list(sim.state))

                atDest, frac = mt.Tree.is_state_at_dest(destination, sim.prevState, sim.state)

                while (not atDest) \
                        and (not mt.Tree.is_state_terminal(sim, sim.state)):
                    dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                    sim.doStep(action)

                    if plot:
                        traj.append(list(sim.state))

                    atDest, frac = mt.Tree.is_state_at_dest(destination, sim.prevState, sim.state)

                if atDest:
                    finalTime = sim.times[sim.state[0]] - (1 - frac)
                    arrivaltimes.append(finalTime)

                if plot:
                    trajsofsim[jj][:len(traj)] = traj
                    buff = traj[-1]
                    fillstates = [[kk] + buff[1:] for kk in range(len(traj), len(sim.times))]
                    if fillstates:
                        trajsofsim[jj][len(traj):] = fillstates
                    traj = []

            if plot:
                if arrivaltimes:
                    minarrivaltimes.append(min(arrivaltimes))
                else:
                    print("Scenario num : " + str(ii) + " did not reach destination")

                meantrajs.append(np.mean(trajsofsim, 0))
                trajsofsim = np.full((ntra, len(sims[0].times), 3), stateinit)
                arrivaltimes = []

        if plot:
            timemin = min(minarrivaltimes)
            basemap_dest = sims[0].prepareBaseMap(proj='aeqd', centerOfMap=stateinit[1:])
            plt.title('Mean initialization trajectory for distance estimation')
            colors = plt.get_cmap("tab20")
            colors = colors.colors[:len(sims)]
            xd, yd = basemap_dest(destination[1], destination[0])
            xs, ys = basemap_dest(stateinit[2], stateinit[1])

            basemap_dest.scatter(xd, yd, zorder=0, c="red", s=100)
            plt.annotate("destination", (xd, yd))
            basemap_dest.scatter(xs, ys, zorder=0, c="green", s=100)
            plt.annotate("start", (xs, ys))

            for ii, sim in enumerate(sims):
                sim.plotTraj(meantrajs_dest[ii], basemap_dest, color=colors[ii], label="Scen. num : " + str(ii))
            plt.legend()

            basemap_time = sims[0].prepareBaseMap(proj='aeqd', centerOfMap=stateinit[1:])
            plt.title('Mean trajectory for minimal travel time estimation')
            basemap_time.scatter(xd, yd, zorder=0, c="red", s=100)
            plt.annotate("destination", (xd, yd))
            basemap_time.scatter(xs, ys, zorder=0, c="green", s=100)
            plt.annotate("start", (xs, ys))

            for ii, sim in enumerate(sims):
                sim.plotTraj(meantrajs[ii], basemap_time, color=colors[ii], label="Scen. num : " + str(ii))

            plt.legend()



        else:
            timemin = min(arrivaltimes)

        #todo maintenant qu on a les mean trajs il faut faire les plots depart, arrivee, mean traj par scenario

        return [destination, timemin]

        # @staticmethod
