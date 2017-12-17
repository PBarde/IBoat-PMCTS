import master as ms
import myTree as mt

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

    def launch_search(self, root_state):

        # Create the threads
        # TODO vérifier qu'il n'y a pad besoin de thread pour le master comme c'est les workers qui appele la fonction du master
        # master_thread = th.Thread(name='master', target=self.master.update) ?? inutile?
        worker_thread = dict()
        for worker in self.workers:
            worker_thread[worker.id] = th.Thread(name='worker' + str(worker.id), target=worker.uct_search,
                                                 args=root_state)

        # Launch the threads
        # master_thread.start()
        for w_th in worker_thread:
            w_th.start()
        # TODO regarder s'il faut utliser la méthode lock() pour blocker les autres threads pendant que le master
        # todo se met à jour par exemple

        # Wait for threads to terminate
        # master_thread.join()
        for w_th in worker_thread:
            w_th.join()
