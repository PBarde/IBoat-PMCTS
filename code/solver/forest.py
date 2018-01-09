import master as ms
import worker as mt
import sys
sys.path.append('../model/')
from weatherTLKT import Weather

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
                          modelcycle = range(1, 21)):

        pathToSaveObj = []
        weather_scen = []
        for ii in modelcycle:

            if ii < 10:
                cycle = '0' + str(ii)
            else:
                cycle = str(ii)

            url = (website + 'gens/gens' + mydate + '/gep' + cycle + '_00z')
            pathToSaveObj.append(('../data/' + mydate + '_gep_' + cycle + '00z.obj'))

            weather_scen.append(Weather.load(pathToSaveObj[ii-1]))

        return weather_scen