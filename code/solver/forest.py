import worker as mt
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from multiprocessing import Process
from master_node import MasterNode
from multiprocessing import Manager

sys.path.append('../model/')
from weatherTLKT import Weather
from simulatorTLKT import Simulator, HOURS_TO_DAY


class Forest:
    """
    Object coordinating a MasterTree and its WorkerTrees during a parallel MCTS search.

    :ivar listsimulators: a list of :class:`simulatorTLKT.Simulator` objects, each simulator should correspond to a different weather scenario
    :vartype listsimulators: list

    :ivar destination: [lat,lon], destination to reach
    :vartype destination: list(float,float)

    :ivar timemin: reference time to define the reward model, comes from the :func:`initialize_simulators` function
    :vartype timemin: float

    :ivar budget: number of nodes in each worker at the end of the search
    :vartype budget: int

    """

    def __init__(self, listsimulators=[], destination=[], timemin=0, budget=100):
        """
        Class constructor

        """

        self.workers = dict()
        nscenario = len(listsimulators)
        self.nscenario = nscenario
        for i, sim in enumerate(listsimulators):
            self.workers[i] = mt.Tree(workerid=i, nscenario=nscenario, ite=0, budget=budget,
                                      simulator=deepcopy(sim), destination=deepcopy(destination), TimeMin=timemin)

    def launch_search(self, root_state, frequency):
        """
        Launches a parallel MCTS search. It uses multiprocessing class. Each worker is a :class:`multiprocessing.Process`
        object. The master is a :class:`multiprocessing.Manager` hosting a dict() (called Master_nodes)
        that can be read and written into by each worker.

        :param root_state: [time index, lat, lon], state of the root node of the sea
        :paramtype root_state: list(int,float,float)

        :param int frequency: number of node that are expanded in a worker before this one updates the corresponding nodes in the master.

        :return: dictionnary of :class:`master_node.MasterNode` the keys are the nodes' hashes. These results must be
                deep copied with :func:`master_node.deepcopy_dict` if you want to do anything with them.
        :rtype: :class:`multiprocessing.Manager.dict`
        """

        # Create the manager
        m = Manager()
        Master_nodes = m.dict()
        Master_nodes[hash(tuple([]))] = MasterNode(len(self.workers), nodehash=hash(tuple([])))

        # Create the workers Process
        worker_process = dict()
        for worker in self.workers.values():
            worker_process[worker.id] = Process(name='worker' + str(worker.id), target=worker.uct_search,
                                                args=(deepcopy(root_state), frequency, Master_nodes))
        # Launch the threads
        for w_p in worker_process.values():
            w_p.start()

        # Wait for process to complete
        for w_th in worker_process.values():
            w_th.join()

        for worker in self.workers.values():
            print("Number of iterations for worker " + str(worker.id) + ": " + str(worker.ite))
        print(id(Master_nodes))
        return Master_nodes


def download_scenarios(mydate, latBound=[43, 50], lonBound=[-10 + 360, 360],
                       website='http://nomads.ncep.noaa.gov:9090/dods/',
                       scenario_ids=range(1, 21)):
    """
    To download the weathers scenarios for a MCTS ** /!\ : needs to be launched in the terminal**.
    Scenarios are saved as '../data/' + mydate + '_gep_' + s_id + '00z.obj' where s_id is the scenario id. They are
    stored as :class:`weatherTLKT.Weather` objects.

    :param string mydate: 'yyyymmdd' date of the day starting the weather forecast in US format
    :param list(float,float) latBound: [latmin, latmax], latitude boundaries of the domain on which the forecast is desired
    :param list(float,float) lonBound: [lonmin, lonmax], longitude boundaries of the domain on which the forecast is desired
    :param string website: url of the server hosting the forecasts
    :param range scenario_ids: range covering the forecasts scenarios we want to download
    """
    pathToSaveObj = []
    for ii in scenario_ids:

        if ii < 10:
            s_id = '0' + str(ii)
        else:
            s_id = str(ii)

        url = (website + 'gens/gens' + mydate + '/gep' + s_id + '_00z')
        pathToSaveObj.append(('../data/' + mydate + '_gep_' + s_id + '00z.obj'))

        Weather.download(url, pathToSaveObj[ii - 1], latBound=latBound, lonBound=lonBound, timeSteps=[0, 64],
                         ens=True)


def load_scenarios(mydate, scenario_ids=range(1, 21), latBound=[-90, 90],
                   lonBound=[0, 360], timeSteps=[0, 64]):
    """
    Loads previously downloaded scenarios and returns the corresponding list of :class:`weatherTLKT.Weather` objects.

    :param string mydate: 'yyyymmdd' date of the day starting the weather forecast in US format
    :param scenario_ids: range covering the forecasts scenarios we want to download
    :param list(float,float) latBound: [latmin, latmax], latitude boundaries of the domain on which the forecast is desired.
                                        If smaller than domain of stored object, returns a cropped object.
                                        Returns stored object otherwise
    :param list(float,float) lonBound: [lonmin, lonmax], longitude boundaries of the domain on which the forecast is desired.
                                        If smaller than domain of stored object, returns a cropped object.
                                        Returns stored object otherwise
    :param list(int,int) timeSteps: [min_t_index, max_t_index], time span of the desired forecast.
                                        If smaller than domain of stored object, returns a cropped object.
                                        Returns stored object otherwise

    :return: List of of :class:`weatherTLKT.Weather`
    :rtype: list()
    """
    pathToSaveObj = []
    weather_scen = []
    for ii in scenario_ids:

        if ii < 10:
            s_id = '0' + str(ii)
        else:
            s_id = str(ii)

        pathToSaveObj.append(('../data/' + mydate + '_gep_' + s_id + '00z.obj'))

        weather_scen.append(Weather.load(pathToSaveObj[ii - 1], latBound, lonBound, timeSteps))

    return weather_scen


def create_simulators(weathers, numberofsim, simtimestep=6, stateinit=(0, 47.5, -3.5 + 360),
                      ndaysim=8, deltalatlon=0.5):
    """
    Creates and returns a list of :class:`simulatorTLKT.Simulator`.

    :param list() weathers: List of :class:`weatherTLKT.Weather` objects, each one is the wind scenario of the
                            :class:`simulatorTLKT.Simulator` object to be created

    :param int numberofsim: number of simulator we want to create. Must be lower or equal to len(weathers). The simulators
                            are created by chronologically browsing the weathers list

    :param float simtimestep: time step in hours of the simulator

    :param list(int,float,float) stateinit: [t_index, lat, lon], initial state of the simulators

    :param int ndaysim: time horizon of the search. Must be smaller or equal to the weather object time horizon.

    :param float deltalatlon: latitude and longitude discretization of the simulators. Just for plotting purposes

    :return: List of :class:`simulatorTLKT.Simulator`
    :rtype: list()
    """
    sims = []
    for jj in range(numberofsim):
        # We shift the times so that all times are in the correct bounds for interpolations
        weathers[jj].time = weathers[jj].time - weathers[jj].time[0]

        # We set up the parameters of the simulation
        times = np.arange(0, ndaysim, simtimestep * HOURS_TO_DAY)
        lats = np.arange(weathers[jj].lat[0], weathers[jj].lat[-1], deltalatlon)
        lons = np.arange(weathers[jj].lon[0], weathers[jj].lon[-1], deltalatlon)
        sims.append(Simulator(times, lats, lons, weathers[jj], list(stateinit)))

    return sims


def play_multiple_scenarios(sims):
    """
    Display an interactive representation of the wind conditions of a list of :class:`simulatorTLKT.Simulator` objects.

    :param list() sims: List of the :class:`simulatorTLKT.Simulator` objects
    """

    # Create the workers Process
    player_process = []
    for ii, sim in enumerate(sims):
        player_process.append(Process(target=sim.play_scenario, args=[ii]))
    # Launch the threads
    for w_p in player_process:
        w_p.start()

    # Wait for process to complete
    for w_th in player_process:
        w_th.join()


def initialize_simulators(sims, ntra, stateinit, missionheading, plot=False):
    """
    Find the destination point and the reference time for a list of simulators. These quantities are mandatory in order
    to carry out a MCTS parallel search.

    :param list() sims: List of :class:`simulatorTLKT.Simulator`
    :param int ntra: Number of trajectories used to estimate the initialization
    :param list(int,float,float) stateinit: t_index, lat, lon], starting point of the MCTS search
    :param float missionheading: angle in deg wrt true North of the desired destination from stateinit
    :param bool plot: if True displays the initialization trajectories
    :return: [destination, timemin], destination are the coordinates of the destination point and timemin is in days
    :rtype: list(list(float,float), float)
    """

    meanarrivaldistances = []

    if plot:
        meantrajs_dest = []
        trajsofsim = np.full((ntra, len(sims[0].times), 3), stateinit)
        traj = []

    for sim in sims:
        arrivaldistances = []

        for ii in range(ntra):
            sim.reset(stateinit)

            if plot:
                traj.append(list(sim.state))

            for t in sim.times[0:-1]:

                if not mt.Tree.is_state_terminal(sim, sim.state):
                    sim.doStep(missionheading)
                else: break

                if plot:
                    traj.append(list(sim.state))

            if plot:
                trajsofsim[ii][:len(traj)] = traj
                buff = traj[-1]
                fillstates = [[kk] + buff[1:] for kk in range(len(traj), len(sim.times))]
                if fillstates:
                    trajsofsim[ii][len(traj):] = fillstates
                traj = []

            dist, dump = sim.getDistAndBearing(stateinit[1:], (sim.state[1:]))
            arrivaldistances.append(dist)

        meanarrivaldistances.append(np.mean(arrivaldistances))
        if plot:
            meantrajs_dest.append(np.mean(trajsofsim, 0))
            trajsofsim = np.full((ntra, len(sims[0].times), 3), stateinit)

    mindist = np.min(meanarrivaldistances)
    destination = sim.getDestination(mindist, missionheading, stateinit[1:])

    if plot:
        minarrivaltimes = []
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
                finalTime = sim.times[sim.state[0]] - \
                            (1 - frac)*(sim.times[sim.state[0]] - sim.times[sim.state[0] - 1])
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
        basemap_dest = sims[0].prepareBaseMap()
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

        basemap_time = sims[0].prepareBaseMap()
        plt.title('Mean trajectory for minimal travel time estimation')
        basemap_time.scatter(xd, yd, zorder=0, c="red", s=100)
        plt.annotate("destination", (xd, yd))
        basemap_time.scatter(xs, ys, zorder=0, c="green", s=100)
        plt.annotate("start", (xs, ys))

        for ii, sim in enumerate(sims):
            sim.plotTraj(meantrajs[ii], basemap_time, color=colors[ii], label="Scen. num : " + str(ii))

        plt.legend()
        plt.show()

    else:
        timemin = min(arrivaltimes)

    return [destination, timemin]

