#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 13:48:13 2017

@author: j.belley
"""

import sys
sys.path.append("../solver")
#from MyTree import Tree
from worker import Tree
import numpy as np
from math import atan2


#Simulator.Boat.Uncertainitycoeff=0

# PB = delta_S à corriger (fait par trigo simple)
# attention calcul du temps total?

class Node():
    
    def __init__(self,time,lat,lon,parent=None,action=None,C=None,D=None):
        self.time=time #indice
        self.lat=lat
        self.lon=lon
        self.pere=parent
        self.act=action
        self.C=C
        self.D=D
        
        
    def give_state(self):
        return [self.time,self.lat,self.lon]
    
    def __str__(self):
        return '({},{}) à {}\nC={},D={}'.format(self.lat,self.lon,self.time,self.C,self.D)
    
    
class Secteur():
    
    def __init__(self,cap_sup,cap_inf):
        self.cap_sup = cap_sup
        self.cap_inf = cap_inf
        self.liste_noeud = [] # attention s'assurer dans la construction de la correspondance entre les indices des 2 listes
        self.liste_distance = []
        
    def recherche_meilleur_noeud(self):
        try:
            meilleur_noeud = self.liste_noeud[self.liste_distance.index(max(self.liste_distance))]
            return meilleur_noeud
        except ValueError:
            return None
    
    def __str__(self):
        return '({},{}) à {}'.format(self.cap_inf,self.cap_sup,self.liste_distance)
    

class Isochrone():
    
    def __init__(self,simulateur,coord_depart,coord_arrivee,delta_cap=10,increment_cap=9,nb_secteur=10,resolution=200):

        self.sim=simulateur
        self.dep=coord_depart   #liste des coord de départ (lat,lon)
        self.arr=coord_arrivee  #liste des coord d'arrivée (lat,lon)
        noeuddep=Node(0,coord_depart[0],coord_depart[1]) #attention si référence par parentée n'empêche pas le rammase miette de le supprimer
        self.isochrone_actuelle=[noeuddep]
        self.isochrone_future=[]
        self.distance_moy_iso = 0
        self.reso = resolution
        self.p = nb_secteur
        self.constante = np.pi/(60*180)
        self.delta_t = (self.sim.times[noeuddep.time+1]-self.sim.times[noeuddep.time])
        self.liste_actions = []
        self.liste_positions = []
        self.temps_transit = 0
        for l in range(-increment_cap,increment_cap+1):
            self.liste_actions.append(l*delta_cap)
        D0,C0 = self.sim.getDistAndBearing(self.dep,self.arr)
        self.C0 = C0
        C = []
        for action in self.liste_actions:
            C.append(self.recentrage_cap(C0+action))
        self.isochrone_actuelle=[]
        for cap in C:
            self.sim.reset(noeuddep.give_state())
            state = self.sim.doStep(cap)
            D1,C1 = self.sim.getDistAndBearing(self.dep,state[1:3])
            current_node = Node(state[0],state[1],state[2],noeuddep,cap,C1,D1)
            self.isochrone_actuelle.append(current_node)
        
    
    def recentrage_cap(self,cap):
        return cap%360
    
        
    def reset(self,coord_depart=None,coord_arrivee=None):
        if (not coord_depart==None):
            self.dep=coord_depart   #liste des coord de départ (lat,lon)
        if (not coord_arrivee==None):
            self.arr=coord_arrivee  #liste des coord d'arrivée (lat,lon)
        
        noeuddep=Node(0,self.dep[0],self.dep[1])
        self.isochrone_actuelle=[noeuddep]
        self.isochrone_future=[]
        self.distance_moy_iso = 0
        self.liste_positions = []
        self.temps_transit = 0
        C = []
        for action in self.liste_actions:
            C.append(self.recentrage_cap(self.C0+action))
        self.isochrone_actuelle=[]
        for cap in C:
            self.sim.reset(noeuddep.give_state())
            state = self.sim.doStep(cap)
            D1,C1 = self.sim.getDistAndBearing(self.dep,state[1:3])
            current_node = Node(state[0],state[1],state[2],noeuddep,cap,C1,D1)
            self.isochrone_actuelle.append(current_node)
        return None

    def isochrone_brouillon(self):
        self.isochrone_future=[]
        compteur = 0
        self.distance_moy_iso=0
        for x_i in self.isochrone_actuelle:
            C = []
            for action in self.liste_actions:
                C.append(self.recentrage_cap(x_i.C+action))
            for cap in C:
                self.sim.reset(x_i.give_state())
                state = self.sim.doStep(cap)
                Dij,Cij = self.sim.getDistAndBearing(self.dep,state[1:3])
                futur_node = Node(state[0],state[1],state[2],x_i,cap,Cij,Dij)
                self.isochrone_future.append(futur_node)
                self.distance_moy_iso += Dij
                compteur += 1
        self.distance_moy_iso = self.distance_moy_iso/compteur
        return None
    
    def secteur_liste(self):
        #delta_S = self.constante*self.reso/np.sin(self.constante*self.distance_moy_iso) #pb définition delta_S
        delta_S = atan2(self.reso,self.distance_moy_iso)*180/np.pi
        liste_S = []
        for k in range(2*self.p):
            k+=1
            cap_sup = self.recentrage_cap(self.C0+(k-self.p)*delta_S)
            cap_inf = self.recentrage_cap(self.C0+(k-self.p-1)*delta_S)
            liste_S.append(Secteur(cap_sup,cap_inf))
        return liste_S,delta_S
    
    def associer_xij_a_S(self,liste_S,delta_S):
        for xij in self.isochrone_future:
            Cij = xij.C
            borne_sup = liste_S[-1].cap_sup
            borne_inf = liste_S[0].cap_inf
            if (Cij < borne_inf or Cij > borne_sup):
                pass
            else:
                if (Cij-self.C0) <= -180:
                    diff = (Cij-self.C0)+360
                elif (Cij-self.C0) >=180:
                    diff = (Cij-self.C0)-360
                else:
                    diff = (Cij-self.C0)
                indice_S = int(diff/delta_S + self.p)
                liste_S[indice_S].liste_noeud.append(xij)
                liste_S[indice_S].liste_distance.append(xij.D)
        return liste_S
        
    def nouvelle_isochrone_propre(self,liste_S):
        self.isochrone_actuelle = []
        for Sect in liste_S:
            noeud_a_garder = Sect.recherche_meilleur_noeud()
            if noeud_a_garder == None:
                pass
            else:
                self.isochrone_actuelle.append(noeud_a_garder)
        return None
    
    def isochrone_proche_arrivee(self):
        Top_noeud = []
        Top_dist = []
        Top_cap = []
        arrive = False
        for xi in self.isochrone_actuelle:
            Ddest,Cdest = self.sim.getDistAndBearing([xi.lat,xi.lon],self.arr)
            if Ddest <= 10000:
                Top_noeud.append(xi)
                Top_dist.append(Ddest)
                Top_cap.append(Cdest)
                arrive = True
        return arrive,Top_noeud,Top_dist,Top_cap
        
    def aller_point_arrivee(self,Top_noeud,Top_dist,Top_cap):
        Top_time = []
        for i in range(len(Top_noeud)):
            atDest = False
            frac = 0
            noeud_final = Top_noeud[i]
            cap_a_suivre = Top_cap[i]
            self.sim.reset([noeud_final.time,noeud_final.lat,noeud_final.lon])
            while (not atDest):
                self.sim.doStep(cap_a_suivre)
                atDest,frac =Tree.is_state_at_dest(self.arr,self.sim.prevState,self.sim.state)
            #temps_total = self.sim.times[self.sim.state[0]]-(1-frac)
            temps_total = self.sim.times[self.sim.state[0]] + frac*self.delta_t
            Top_time.append(temps_total)
        indice_solution = Top_time.index(min(Top_time))
        meilleur_noeud_final = Top_noeud[indice_solution]
        temps_total = Top_time[indice_solution]
        cap_final = Top_cap[indice_solution]
        return meilleur_noeud_final,temps_total,cap_final
        
    def isochrone_methode(self):
        temps_total = 0
        liste_point_passage = []
        liste_de_caps_solution = []
        arrive = False
        while (not arrive):
            self.isochrone_brouillon()
            liste_S,delta_S = self.secteur_liste()
            liste_S = self.associer_xij_a_S(liste_S,delta_S)
            self.nouvelle_isochrone_propre(liste_S)
            print(self.isochrone_actuelle[0])
            arrive,Top_noeud,Top_dist,Top_cap = self.isochrone_proche_arrivee()
            print(arrive)
            #pour chaque noeud Top faire simu jusqu'à isstateatdest et calculer temps pour discriminer le meilleur noeud
            #remonter les noeuds parents
        meilleur_noeud_final,temps_total,cap_final = self.aller_point_arrivee(Top_noeud,Top_dist,Top_cap)
        liste_de_caps_solution.append(cap_final)
        while meilleur_noeud_final.pere is not None:
            liste_point_passage.append([meilleur_noeud_final.lat,meilleur_noeud_final.lon])
            liste_de_caps_solution.append(meilleur_noeud_final.act)
            meilleur_noeud_final = meilleur_noeud_final.pere
        liste_point_passage.append([meilleur_noeud_final.lat,meilleur_noeud_final.lon])
        
        self.liste_positions = liste_point_passage[::-1]
        self.liste_positions.append(self.arr)
        self.liste_actions = liste_de_caps_solution[::-1]
        self.temps_transit = temps_total
        
        return self.temps_transit,self.liste_actions,self.liste_positions
        