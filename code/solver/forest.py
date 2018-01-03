import master as ms
import worker as mt

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
            self.workers[i] = mt.Tree(master=self.master, workerid=i, ite=0, budget=1000,
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

        # Launch the threads
        for w_th in worker_thread.values():
            w_th.start()

        # Launch the master in this thread
        self.master.update(self.workers, worker_events, end_events)

        # Wait for threads to complete
        for w_th in worker_thread.values():
            w_th.join()
