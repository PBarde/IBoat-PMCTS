#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:48:13 2017

@author: j.belley
"""

import sys

sys.path.append("../solver")
# from MyTree import Tree
from worker import Tree
import numpy as np
from math import atan2
import matplotlib.pyplot as plt
from scipy.special import erf


# Simulator.Boat.Uncertainitycoeff=0

# PB = delta_S à corriger (fait par trigo simple)

class Node():
    """ Class which is used to make the points of the isochrone curves. \
    It is based on a node structure of a linked list. 
    
    :ivar int time: time indice used by the ship simulator to define a state.
    
    :ivar float lat: latitude used by the ship simulator to define a state.
    
    :ivar float lon: longitude used by the ship simulator to define a state.
    
    :ivar isochrones.Node pere: parent node from the previous isochrone curve.
    
    :ivar float act: heading follow from the parent node to the actual node.
    
    :ivar float C: heading between the starting point and the node position.
    
    :ivar float D: distance between the starting point and the node position.
    
    """

    def __init__(self, time, lat, lon, parent=None, action=None, C=None, D=None):
        """
        Class constructor
        """

        self.time = time  # indice
        self.lat = lat
        self.lon = lon
        self.pere = parent
        self.act = action
        self.C = C
        self.D = D

    def give_state(self):
        """ Method which returns the state of a node. \
        The definition of a state is given in simulatorTLKT documentation"""
        return [self.time, self.lat, self.lon]

    def __str__(self):
        return '({},{}) à {}\nC={},D={}'.format(self.lat, self.lon, self.time, self.C, self.D)


class Secteur():
    """ Class used to partition the space in order to apply Hagiwara's isochrone method. \
    
    :ivar float cap_sup: upper bond of the angular sector which start from the beginning point.
    
    :ivar float cap_inf: lower bond of the angular sector which start from the beginning point.
    
    :ivar list liste_noeud: list of the nodes which belong to the angular sector.
    
    :ivar list liste_distance: list of the distances between the starting point and the nodes in liste_noeud.
    
    """

    def __init__(self, cap_sup, cap_inf):

        """
        Class constructor
        """

        self.cap_sup = cap_sup
        self.cap_inf = cap_inf
        self.liste_noeud = []  # attention s'assurer dans la construction de la correspondance entre les indices des 2 listes
        self.liste_distance = []

    def recherche_meilleur_noeud(self):

        """ Method which returns the farthest node of liste_noeud according to the data in liste_distance """

        try:
            meilleur_noeud = self.liste_noeud[self.liste_distance.index(max(self.liste_distance))]
            return meilleur_noeud
        except ValueError:
            return None

    def __str__(self):
        return '({},{}) à {}'.format(self.cap_inf, self.cap_sup, self.liste_distance)


class Isochrone():
    """
    Class making the determinist isochrone method described by Hideki Hagiwara.

    :ivar Simulator sim: Object Simulator which have to be initialized \
        with the geographic position of the problem and the weather forcast desired.

    :ivar list dep: Latitude and longitude in degree of the starting \
        point. For instance, [47.5, 356.5].

    :ivar list arr: Latitude and longitude in degree of the destination \
        point. For instance, [47.8, 352.3].

    :ivar list isochrone_actuelle: List of the nodes which constitute the \
        current isochrone shape.

    :ivar list isochrone_future: List of the nodes which are made in order \
        to find the shape of the next isochrone.
        
    :ivar list isochrone_stock: List of used isochrone (list of list of states)
        
    :ivar float distance_moy_iso: Average distance of the current isochrone from \
        the starting point in meters.
        
    :ivar float reso: Resolution of the current isochrone (average distance \
        between nodes of the isochrone shape in meters).
    
    :ivar int p: Half of the number of sectors used by the isochrone algorithm \
        (half of the maximal number of nodes of the current isochrone).
        
    :ivar float constante: Used to change units.
    
    :ivar float delta_t: Time step of the simulator in days.
    
    :ivar list liste_actions: List of the bearings in degree that the ship can \
        use to join its destination. After solving the problem, it become the \
        policy to follow to reach the destination.
        
    :ivar list liste_positions: List of the positions (lat,lon) where the ship \
        goes and changes its bearing in order to reach the destination.
        
    :ivar float temps_transit: Time spend during the trip in days.
    
    :ivar float C0: Bearing between the departure and the arrival in degree.
    
    :ivar int temps_dep: time indice used in the ship Simulator. Chose another value than 0 \
        if you want to start your isochrone research at a precise temps (different from forcast \
        initial hour).

    """

    def __init__(self, simulateur, coord_depart, coord_arrivee, delta_cap=10, increment_cap=9, nb_secteur=10,
                 resolution=200, temps=0):

        """
        Class constructor
        delta_cap is the number of degree between two possible actions
        increment_cap is the number of actions allowed on the right of last heading and on the left too.
        """

        self.sim = simulateur
        self.dep = coord_depart[1:]  # liste des coord de départ (lat,lon)
        self.arr = coord_arrivee  # liste des coord d'arrivée (lat,lon)
        self.temps_dep = temps
        noeuddep = Node(temps, coord_depart[1], coord_depart[
            2])
        self.isochrone_actuelle = [noeuddep]
        self.isochrone_future = []
        self.distance_moy_iso = 0
        self.reso = resolution
        self.p = nb_secteur
        self.constante = np.pi / (60 * 180)
        self.delta_t = (self.sim.times[noeuddep.time + 1] - self.sim.times[noeuddep.time])
        self.liste_actions = []
        self.liste_positions = []
        self.isochrone_stock = []
        self.temps_transit = 0
        for l in range(-increment_cap, increment_cap + 1):
            self.liste_actions.append(l * delta_cap)
        D0, C0 = self.sim.getDistAndBearing(self.dep, self.arr)
        self.C0 = C0
        C = []
        for action in self.liste_actions:
            C.append(self.recentrage_cap(C0 + action))
        self.isochrone_actuelle = []
        liste_etats = []
        for cap in C:
            self.sim.reset(noeuddep.give_state())
            state = self.sim.doStep(cap)
            D1, C1 = self.sim.getDistAndBearing(self.dep, state[1:3])
            current_node = Node(state[0], state[1], state[2], noeuddep, cap, C1, D1)
            self.isochrone_actuelle.append(current_node)
            liste_etats.append(current_node.give_state())
        self.isochrone_stock.append(liste_etats)

    def recentrage_cap(self, cap):
        """
        Keep the heading in degree between 0 and 360.
        
        :param float cap: Bearing to correct.
        """
        return cap % 360

    def reset(self, coord_depart=None, coord_arrivee=None):

        """
        Reset the problem to slove with a different departure and arrival but \
        the same weather forcast. To change the weather forcast, change self.sim \
        and reset the Isochrone.
        
        :param list coord_depart: Latitude and longitude in degree of the starting \
        point.
        
        :param list coord_arrivee: Latitude and longitude in degree of the destination \
        point.
        """

        if (not coord_depart == None):
            self.dep = coord_depart  # liste des coord de départ (lat,lon)
        if (not coord_arrivee == None):
            self.arr = coord_arrivee  # liste des coord d'arrivée (lat,lon)

        noeuddep = Node(0, self.dep[0], self.dep[1])
        self.isochrone_actuelle = [noeuddep]
        self.isochrone_future = []
        self.distance_moy_iso = 0
        self.liste_positions = []
        self.isochrone_stock = []
        self.temps_transit = 0
        D0, C0 = self.sim.getDistAndBearing(self.dep, self.arr)
        self.C0 = C0
        C = []
        for action in self.liste_actions:
            C.append(self.recentrage_cap(self.C0 + action))
        self.isochrone_actuelle = []
        for cap in C:
            self.sim.reset(noeuddep.give_state())
            state = self.sim.doStep(cap)
            D1, C1 = self.sim.getDistAndBearing(self.dep, state[1:3])
            current_node = Node(state[0], state[1], state[2], noeuddep, cap, C1, D1)
            self.isochrone_actuelle.append(current_node)
        return None

    def isochrone_brouillon(self):

        """
        Generates all the future nodes reachable from the current isochrone by \
        doing the different actions allowed.
        
        """

        self.isochrone_future = []
        compteur = 0
        self.distance_moy_iso = 0
        for x_i in self.isochrone_actuelle:
            C = []
            for action in self.liste_actions:
                C.append(self.recentrage_cap(x_i.C + action))
            for cap in C:
                self.sim.reset(x_i.give_state())
                state = self.sim.doStep(cap)
                Dij, Cij = self.sim.getDistAndBearing(self.dep, state[1:3])
                futur_node = Node(state[0], state[1], state[2], x_i, cap, Cij, Dij)
                self.isochrone_future.append(futur_node)
                self.distance_moy_iso += Dij
                compteur += 1
        self.distance_moy_iso = self.distance_moy_iso / compteur
        return None

    def secteur_liste(self):

        """
        Creates all the angular sectors used to select the nodes of the future \
        isochrone and return a list of them. Returns also the width of a sector \
        at the average distance of the future isochrone.
        
        """

        # delta_S = self.constante*self.reso/np.sin(self.constante*self.distance_moy_iso) #pb définition delta_S
        delta_S = atan2(self.reso, self.distance_moy_iso) * 180 / np.pi
        liste_S = []
        for k in range(2 * self.p):
            k += 1
            cap_sup = self.recentrage_cap(self.C0 + (k - self.p) * delta_S)
            cap_inf = self.recentrage_cap(self.C0 + (k - self.p - 1) * delta_S)
            liste_S.append(Secteur(cap_sup, cap_inf))
        return liste_S, delta_S

    def associer_xij_a_S(self, liste_S, delta_S):

        """
        Associates to all the nodes reachable from the current isochrone an \
        angular sector and returns the list of those sectors.
        
        """

        for xij in self.isochrone_future:
            Cij = xij.C
            borne_sup = liste_S[-1].cap_sup
            borne_inf = liste_S[0].cap_inf
            if borne_sup > borne_inf:
                if (Cij < borne_inf or Cij > borne_sup):
                    pass
                else:
                    if (Cij - self.C0) <= -180:
                        diff = (Cij - self.C0) + 360
                    elif (Cij - self.C0) >= 180:
                        diff = (Cij - self.C0) - 360
                    else:
                        diff = (Cij - self.C0)
                    indice_S = int(diff / delta_S + self.p)
                    liste_S[indice_S].liste_noeud.append(xij)
                    liste_S[indice_S].liste_distance.append(xij.D)
            else:
                if (0 <= Cij < borne_sup or 360 > Cij > borne_inf):
                    if (Cij - self.C0) <= -180:
                        diff = (Cij - self.C0) + 360
                    elif (Cij - self.C0) >= 180:
                        diff = (Cij - self.C0) - 360
                    else:
                        diff = (Cij - self.C0)
                    indice_S = int(diff / delta_S + self.p)
                    liste_S[indice_S].liste_noeud.append(xij)
                    liste_S[indice_S].liste_distance.append(xij.D)
                else:
                    pass
        return liste_S

    def nouvelle_isochrone_propre(self, liste_S):

        """
        Keep, for each sector, only the farthest node reachable from the current \
        isochrone and so create the new current isochrone.
        
        """
        liste_etats = []
        self.isochrone_actuelle = []
        for Sect in liste_S:
            noeud_a_garder = Sect.recherche_meilleur_noeud()
            if noeud_a_garder == None:
                pass
            else:
                self.isochrone_actuelle.append(noeud_a_garder)
                liste_etats.append(noeud_a_garder.give_state())
        self.isochrone_stock.append(liste_etats)
        return None

    def isochrone_proche_arrivee(self):

        """ Method which tests if the nodes of the current isochrone are close from \
        the arrival point (under 20 km). If they are, it returns a boolean True and the \
        list of the node which are almost arrived """

        Top_noeud = []
        arrive = False
        for xi in self.isochrone_actuelle:
            Ddest, Cdest = self.sim.getDistAndBearing([xi.lat, xi.lon], self.arr)
            if Ddest <= 20000:
                Top_noeud.append(xi)
                arrive = True
        return arrive, Top_noeud

    def aller_point_arrivee(self, Top_noeud):

        """ Method which takes all the nodes close from the arrival and make them arrive \
        by going straight ahead. Then it returns the node which joins earlier the destination, \
        the time spend during all the journey between the starting point and the destination and \
        the list of the heandings followed during the last 20 km """

        Top_time = []
        Top_finish_caps = []
        solution = False
        for i in range(len(Top_noeud)):
            atDest = False
            frac = 0
            noeud_final = Top_noeud[i]
            cap_a_suivre = 0
            finish_caps = []
            self.sim.reset([noeud_final.time, noeud_final.lat, noeud_final.lon])
            try:
                while (not atDest):
                    Ddest, cap_a_suivre = self.sim.getDistAndBearing(self.sim.state[1:3], self.arr)
                    finish_caps.append(cap_a_suivre)
                    self.sim.doStep(cap_a_suivre)
                    atDest, frac = Tree.is_state_at_dest(self.arr, self.sim.prevState, self.sim.state)
                temps_total = self.sim.times[self.sim.state[0]] - (1 - frac) * self.delta_t - self.sim.times[self.temps_dep]
                Top_time.append(temps_total)
                Top_finish_caps.append(finish_caps)
                solution = True
            except IndexError:
                pass
        if solution:
            indice_solution = Top_time.index(min(Top_time))
            meilleur_noeud_final = Top_noeud[indice_solution]
            temps_total = Top_time[indice_solution]
            liste_caps_fin = Top_finish_caps[indice_solution]
            return meilleur_noeud_final, temps_total, liste_caps_fin
        else:
            raise IndexError

    def isochrone_methode(self):

        """ If it can, it finds the ship routing solution which minimises the travel \
        time between two points with determinist ship dynamic and weather forcast. \
        It returns the minimum time of trip, the list of heading used during the journey, \
        the list of the final headings and the list of the nodes positions. """

        temps_total = 0
        liste_point_passage = []
        liste_de_caps_solution = []
        arrive = False
        try:

            while (not arrive):
                self.isochrone_brouillon()
                liste_S, delta_S = self.secteur_liste()
                liste_S = self.associer_xij_a_S(liste_S, delta_S)
                self.nouvelle_isochrone_propre(liste_S)
                arrive, Top_noeud = self.isochrone_proche_arrivee()
                # pour chaque noeud Top faire simu jusqu'à isstateatdest et calculer temps pour discriminer le meilleur noeud
                # remonter les noeuds parents
            try:

                meilleur_noeud_final, temps_total, liste_caps_fin = self.aller_point_arrivee(Top_noeud)
                while meilleur_noeud_final.pere is not None:
                    liste_point_passage.append([meilleur_noeud_final.lat, meilleur_noeud_final.lon])
                    liste_de_caps_solution.append(meilleur_noeud_final.act)
                    meilleur_noeud_final = meilleur_noeud_final.pere
                liste_point_passage.append([meilleur_noeud_final.lat, meilleur_noeud_final.lon])

                self.liste_positions = liste_point_passage[::-1]
                self.liste_positions.append(self.arr)
                self.liste_actions = liste_de_caps_solution[::-1]
                self.temps_transit = temps_total

            except IndexError:

                print('Pas de solution trouvée dans le temps imparti.\nVeuillez raffiner vous paramètres de recherche.')
                self.temps_transit = None
                self.liste_actions = None
                liste_caps_fin = None
                self.liste_positions = None

        except IndexError:

            print('Pas de solution trouvée dans le temps imparti.\nVeuillez raffiner vos paramètres de recherche.')
            self.temps_transit = None
            self.liste_actions = None
            liste_caps_fin = None
            self.liste_positions = None

        return self.temps_transit, self.liste_actions, liste_caps_fin, self.liste_positions

    def positions_to_states(self):

        """ Changes the list of nodes' position in a list of states. """

        liste_states = []
        for i in range(len(self.liste_positions)):
            liste_states.append(np.array([i, self.liste_positions[i][0], self.liste_positions[i][1]]))
        return liste_states

        # condition arret temps depasse, recalculer heading finale, nombre de pas temps tout droit, faire visu de la trajectoire


def estimate_perfomance_plan(sims, ntra, stateinit, destination, plan=list(), plot=False, verbose=True):
    """
    Estimates the performances of two plans and compares them on two scenarios.

    :param list() sims: List of :class:`simulatorTLKT.Simulator`
    :param int ntra: Number of trajectories used to estimate the performances on each scenarios
    :param list(int,float,float) stateinit: [t_index, lat, lon], starting point of the plans
    :param list(int,float,float) destination: [t_index, lat, lon], destination point of the plans
    :param list plan: list of actions to apply
    :param bool plot: if True displays the mean trajectories per scenario
    :param bool verbose: if True verbose results
    :return: mean_arrival_times, var_arrival_times, global_mean_time, variance_globale with length : len(list) = len(sims)
    :rtype: list(float), list(float), float, float
    """

    ################### Arrival Time #############################

    meantrajs = []
    mean_arrival_times = []
    var_arrival_times = []
    all_arrival_times = []
    nb_actions = len(plan)
    for _, sim in enumerate(sims):
        arrivaltimes = []
        trajsofsim = np.zeros((ntra, len(sims[0].times), 3))

        for ii in range(ntra):

            traj = []
            sim.reset(stateinit)
            traj.append(list(sim.state))
            compte_action = 0

            while (compte_action < nb_actions):
                action = plan[compte_action]
                compte_action += 1
                sim.doStep(action)
                traj.append(list(sim.state))
                
            if nb_actions == 0:
                dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                sim.doStep(action)
                traj.append(list(sim.state))

            atDest, frac = Tree.is_state_at_dest(destination, sim.prevState, sim.state)

            while (not atDest) \
                    and (not Tree.is_state_terminal(sim, sim.state)):
                dist, action = sim.getDistAndBearing(sim.state[1:], destination)
                sim.doStep(action)
                traj.append(list(sim.state))
                atDest, frac = Tree.is_state_at_dest(destination, sim.prevState, sim.state)

            if atDest:
                finalTime = sim.times[sim.state[0]] - \
                            (1 - frac) * (sim.times[sim.state[0]] - sim.times[sim.state[0] - 1])
                arrivaltimes.append(finalTime)
                all_arrival_times.append(finalTime)
            else:
                finalTime = sim.times[-1]
                arrivaltimes.append(finalTime)
                all_arrival_times.append(finalTime)

            trajsofsim[ii, :, :] = traj[-1]
            trajsofsim[ii, :, 0] = [i for i in range(len(sim.times))]
            trajsofsim[ii, :len(traj), :] = traj

        meantrajs.append(np.mean(trajsofsim, 0))
        average_scenario = np.mean(arrivaltimes)
        mean_arrival_times.append(average_scenario)

        variance_scenario = 0
        for value in arrivaltimes:
            variance_scenario += (average_scenario - value) ** 2
        variance_scenario = variance_scenario / ntra
        var_arrival_times.append(variance_scenario)

    global_mean_time = np.mean(all_arrival_times)
    variance_globale = 0
    for value in all_arrival_times:
        variance_globale += (global_mean_time - value) ** 2
    variance_globale = variance_globale / len(all_arrival_times)

    if plot:

        basemap_time = sims[0].prepareBaseMap(proj='aeqd', centerOfMap=stateinit[1:])
        plt.title('Mean trajectory for minimal travel time estimation')

        colors = plt.get_cmap("tab20")
        colors = colors.colors[:len(sims)]
        xd, yd = basemap_time(destination[1], destination[0])
        xs, ys = basemap_time(stateinit[2], stateinit[1])

        basemap_time.scatter(xd, yd, zorder=0, c="red", s=100)
        plt.annotate("destination", (xd, yd))
        basemap_time.scatter(xs, ys, zorder=0, c="green", s=100)
        plt.annotate("start", (xs, ys))

        for ii, sim in enumerate(sims):
            sim.plotTraj(meantrajs[ii], basemap_time, color=colors[ii], label="Scen. num : " + str(ii))

        plt.legend()

    if verbose:
        for nb in range(len(sims)):
            print("temps scénario isochrones ", nb, " = ", mean_arrival_times[nb])
            print("variance scénario isochrones   = ", var_arrival_times[nb])
            print()
        print("moyenne des temps isochrones                   = ", global_mean_time)
        print("variance globale des isochrones                = ", variance_globale)

    return [global_mean_time] + mean_arrival_times, [variance_globale] + var_arrival_times


def plot_trajectory(sim, trajectoire, quiv=True, scatter=False):
    """ plot the trajectory on the map of the simulator with the wind forcast associated """
    if len(trajectoire[0]) == 2:
        for i, el in enumerate(trajectoire):
            trajectoire[i] = [i] + el

    m = sim.prepareBaseMap(centerOfMap=trajectoire[0][1:], proj='aeqd')
    if len(trajectoire[0]) == 2:
        for i, el in enumerate(trajectoire):
            trajectoire[i] = [i] + el

    sim.plotTraj(trajectoire, m, quiv=quiv, scatter=scatter)
    plt.show()


def plot_comparision(means1, var1, means2, var2, mean_straight_line, var_straight_line, titles):
    """ plot the histograms of the mean times obtained by PMCTS and Isochrones optimal plans and straight line strategy
    for all the weather forcast scenarios. In addition it plots also the standard deviation 
    of each mean time.
    Attention lenght(titles) = 3 """
    N = len(means1)
    
    std1 = np.sqrt(var1)
    std2 = np.sqrt(var2)
    std_straight_line = np.sqrt(var_straight_line)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15  # the width of the bar
    groups = ["Global"]
    for i in range(N - 1):
        groups.append("Sc {}".format(i))

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, means1, width, color='r', yerr=std1)

    rects2 = ax.bar(ind + width, means2, width, color='b', yerr=std2)
    
    rects3 = ax.bar(ind + 2*width, mean_straight_line, width, color='g', yerr=std_straight_line)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean times of arrivals')
    ax.set_title('Means times of arrivals by scenario and strategy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(groups)

    ax.legend((rects1[0], rects2[0], rects3[0]), titles)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.show()


def plot_comparision_percent(means1, var1, means2, var2, mean_straight_line, var_straight_line, titles):
    """ plot the histograms of the mean times obtained by PMCTS and Isochrones optimal plans
    for all the weather forcast scenarios. In addition it plots also the variance 
    of each mean time. Means and variances of the travelling times obtained by the optimal plans
    are normalised by the ones of the straight line plan (in percent).
    Attention lenght(titles) = 2 """
    N = len(means1)
    
    means1 = np.array(means1)
    means2 = np.array(means2)
    mean_straight_line = np.array(mean_straight_line)
    var1 = np.array(var1)
    var2 = np.array(var2)
    var_straight_line = np.array(var_straight_line)
    
    means1_percent = np.divide(means1,mean_straight_line) * 100
    means2_percent = np.divide(means2,mean_straight_line) * 100
    var1_percent = np.divide(var1,var_straight_line) * 100
    var2_percent = np.divide(var2,var_straight_line) * 100
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bar
    groups = ["Global"]
    for i in range(N - 1):
        groups.append("Sc {}".format(i))

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, means1_percent, width, color='r')

    rects2 = ax.bar(ind + width, means2_percent, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean times of arrivals')
    ax.set_title('Means times of arrivals by scenario and strategy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(groups)

    ax.legend((rects1[0], rects2[0]), titles)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()
    
    fig, ax = plt.subplots()
    rects3 = ax.bar(ind, var1_percent, width, color='r')

    rects4 = ax.bar(ind + width, var2_percent, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Variances of times of arrivals')
    ax.set_title('Variances of times of arrivals by scenario and strategy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(groups)

    ax.legend((rects3[0], rects4[0]), titles)

    autolabel(rects3)
    autolabel(rects4)

    plt.show()
    
def plot_mean_squared_error(means1, var1, means2, var2, mean_straight_line, var_straight_line, titles):
    """ plot the histograms of the mean squared error on times obtained by PMCTS and Isochrones optimal plans
    and straight line strategy for all the weather forcast scenarios.
    Attention lenght(titles) = 3 """
    N = len(means1)
    err1 = []
    err2 = []
    err_sl = []
    
    for j in range(N):
        t_optim = min([means1[j], means2[j], mean_straight_line[j]])
        err1.append((t_optim-means1[j])**2 + var1[j])
        err2.append((t_optim-means2[j])**2 + var2[j])
        err_sl.append((t_optim-mean_straight_line[j])**2 + var_straight_line[j])
        
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15  # the width of the bar
    groups = ["Global"]
    for i in range(N - 1):
        groups.append("Sc {}".format(i))

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, err1, width, color='r')

    rects2 = ax.bar(ind + width, err2, width, color='b')
    
    rects3 = ax.bar(ind + 2*width, err_sl, width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Mean squared error on times of arrivals')
    ax.set_title('Mean squared errors on times of arrivals by scenario and strategy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(groups)

    ax.legend((rects1[0], rects2[0], rects3[0]), titles)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.show()
    
def plot_risk_probability(means1, var1, means2, var2, titles, t_lim = 0):
    """ plot the histograms of the probability of times obtained by PMCTS and Isochrones optimal plans
    to be over t_lim for all the weather forcast scenarios.
    Attention lenght(titles) = 2 """
    N = len(means1)
    risk1 = []
    risk2 = []
    
    for j in range(N):
        if t_lim == 0:
            t_bad = max([means1[j], means2[j]])
        else:
            t_bad = t_lim
        u1 = (t_bad - means1[j])/var1[j]
        risk1.append(1-((1/2)*(1 + erf(u1/np.sqrt(2)))))
        u2 = (t_bad - means2[j])/var2[j]
        risk2.append(1-((1/2)*(1 + erf(u2/np.sqrt(2)))))
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bar
    groups = ["Global"]
    for i in range(N - 1):
        groups.append("Sc {}".format(i))

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, risk1, width, color='r')

    rects2 = ax.bar(ind + width, risk2, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Probability of times of arrivals to be over t_lim')
    ax.set_title('Probability of times of arrivals to be over t_lim by scenario and strategy')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(groups)

    ax.legend((rects1[0], rects2[0]), titles)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '%.2f' % height,
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()