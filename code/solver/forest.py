import master as ms
import worker as mt
import sys
sys.path.append('../model/')
from weatherTLKT import Weather
from simulatorTLKT import Simulator, HOURS_TO_DAY
import numpy as np

import threading as th


class Forest:
    """
    Object coordinating a MasterTree and its WorkerTrees
    """

    def __init__(self, listsimulators=[], destination=[], timemin=0):
        # change the constant in master module
        ms.NUMSCENARIOS = len(listsimulators)

        self.master = ms.MasterTree()
        self.workers = dict()
        for i, sim in enumerate(listsimulators):
            self.workers[i] = mt.Tree(master=self.master, workerid=i, ite=0, budget=100,
                                      simulator=sim, destination=destination, TimeMin=timemin)

    def launch_search(self, root_state, frequency):
        # Create the events
        worker_events = dict()  # to notify the master that a buffer is ready
        end_events = dict()  # to tell the master that the search is done
        for worker in self.workers.values():
            worker_events[worker.id] = th.Event()
            end_events[worker.id] = th.Event()

        # Create the workers threads, passing their events in parameter
        worker_thread = dict()
        for worker in self.workers.values():
            worker_thread[worker.id] = th.Thread(name='worker' + str(worker.id), target=worker.uct_search,
                                                 args=(root_state, frequency, worker_events[worker.id],
                                                       end_events[worker.id]))
        # Create the master thread, passing the workers and their events in parameter
        master_thread = th.Thread(name='master', target=self.master.update,
                                  args=(self.workers, worker_events, end_events))
        # Launch the threads
        master_thread.start()
        for w_th in worker_thread.values():
            w_th.start()

        # Wait for threads to complete
        master_thread.join()
        for w_th in worker_thread.values():
            w_th.join()

        for worker in self.workers.values():
            print("Number of iterations for worker " + str(worker.id) + ": " + str(worker.ite))


    @staticmethod
    def download_scenarios(mydate, latBound = [43, 50],lonBound = [-10 + 360, 360],
                          website = 'http://nomads.ncep.noaa.gov:9090/dods/',
                          modelcycle = range(1, 21)):
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

            Weather.download(url, pathToSaveObj[ii-1], latBound=latBound, lonBound=lonBound, timeSteps=[0, 64], ens=True)


    @staticmethod
    def load_scenarios(mydate, website = 'http://nomads.ncep.noaa.gov:9090/dods/',
                          modelcycle = range(1, 21),latBound=[-90, 90],
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

            weather_scen.append(Weather.load(pathToSaveObj[ii-1],latBound, lonBound, timeSteps))

        return weather_scen

    @staticmethod
    def create_simulators(weathers, numberofsim, simtimestep = 6, stateinit = [0, 47.5, -3.5 + 360],
                          ndaysim = 8, delatlatlon = 0.5):
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
    def initialize_simulators(sims,ntra,stateinit,missionheading, plot = False):

        global minarrivaltimes
        arrivalpositions = np.zeros((ntra * len(sims), 2))
        ii = 0

        if plot :
            meantrajs_dest = []
            trajsofsim = []
            traj = []

        for sim in sims:

            for _ in range(ntra):
                sim.reset(stateinit)
                
                if plot : 
                  traj.append(list(sim.state[1:]))

                for t in sim.times[0:-1]:
                    sim.doStep(missionheading)

                    if plot :
                        traj.append(list(sim.state[1:]))

                if plot :
                    trajsofsim.append(list(traj))
                    traj = []

                arrivalpositions[ii, :] = list(sim.state[1:])
                ii += 1

            if plot :
                meantrajs_dest.append(np.mean(trajsofsim,0))
                trajsofsim = []

        latdest = np.mean(arrivalpositions[:, 0])
        londest = np.mean(arrivalpositions[:, 1])
        destination = [latdest, londest]
        
        if plot :
            minarrivaltimes = []
            shape = (len(sims),len(sims[0].times),2)
            meantrajs = np.full(shape,destination)
            trajsofsim = np.full((ntra,len(sims[0].times),2),destination)

        arrivaltimes = []
        
        for ii,sim in enumerate(sims):

            for jj in range(ntra):
                sim.reset(stateinit)
                
                if plot:
                    traj = []
                    traj.append(list(sim.state[1:]))
                    
                dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                sim.doStep(action)

                if plot:
                    traj.append(list(sim.state[1:]))

                atDest, frac = mt.Tree.is_state_at_dest(destination, sim.prevState, sim.state)

                while (not atDest) \
                        and (not mt.Tree.is_state_terminal(sim, sim.state)):
                    dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                    sim.doStep(action)

                    if plot :
                        traj.append(list(sim.state[1:]))

                    atDest, frac = mt.Tree.is_state_at_dest(destination, sim.prevState, sim.state)

                if atDest:
                    finalTime = sim.times[sim.state[0]] - (1 - frac)
                    arrivaltimes.append(finalTime)

                if plot :
                    trajsofsim[jj][:len(traj)] = traj
                    traj = []

            if plot :
              if arrivaltimes :
                  minarrivaltimes.append(min(arrivaltimes))
                  
              meantrajs[ii] = np.mean(trajsofsim,0)
              trajsofsim = np.full((ntra,len(sims[0].times),2),destination)
              arrivaltimes = []

        if plot :
            timemin = min(minarrivaltimes)

        else :
            timemin = min(arrivaltimes)

# todo maintenant qu'on a les mean trajs il faut faire les plots depart, arrivée, mean traj par scenario

        return [destination,timemin,meantrajs, meantrajs_dest]

    # @staticmethod
