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
    
    :ivar int p: Half of the number of sectors used by the ischrone algorithm \
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
        """

        self.sim = simulateur
        self.dep = coord_depart  # liste des coord de départ (lat,lon)
        self.arr = coord_arrivee  # liste des coord d'arrivée (lat,lon)
        self.temps_dep = temps
        noeuddep = Node(temps, coord_depart[0], coord_depart[
            1])  # attention si référence par parentée n'empêche pas le rammase miette de le supprimer
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
        for i in range(len(Top_noeud)):
            atDest = False
            frac = 0
            noeud_final = Top_noeud[i]
            cap_a_suivre = 0
            finish_caps = []
            self.sim.reset([noeud_final.time, noeud_final.lat, noeud_final.lon])
            while (not atDest):
                Ddest, cap_a_suivre = self.sim.getDistAndBearing(self.sim.state[1:3], self.arr)
                finish_caps.append(cap_a_suivre)
                self.sim.doStep(cap_a_suivre)
                atDest, frac = Tree.is_state_at_dest(self.arr, self.sim.prevState, self.sim.state)
            temps_total = self.sim.times[self.sim.state[0]] - (1 - frac) * self.delta_t - self.sim.times[self.temps_dep]
            Top_time.append(temps_total)
            Top_finish_caps.append(finish_caps)
        indice_solution = Top_time.index(min(Top_time))
        meilleur_noeud_final = Top_noeud[indice_solution]
        temps_total = Top_time[indice_solution]
        liste_caps_fin = Top_finish_caps[indice_solution]
        return meilleur_noeud_final, temps_total, liste_caps_fin

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

            print('Pas de solution trouvée dans le temps imparti.\nVeuillez raffiner vous paramètres de recherche.')
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
