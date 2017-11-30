# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:28:31 2017

@author: Jean-Michel
"""
import sys
sys.path.append("../solver")
from MyTree import Tree
#Simulator.Boat.Uncertainitycoeff=0

class Node():
    
    def __init__(self,time,lat,lon,parent=None,action=None,g=0,h=0,f=0):
        self.time=time #indice
        self.lat=lat
        self.lon=lon
        self.pere=parent
        self.act=action
        self.g=g #coût réel
        self.h=h #coût restant estimé / heuristique
        self.f=f #coût total estimé
        
    def give_state(self):
        return [self.time,self.lat,self.lon]
    
    def __str__(self):
        return '({},{}) à {}\ng={},h={},f={}'.format(self.lat,self.lon,self.time,self.g,self.h,self.f)
        
class Pathfinder():
    
    def __init__(self,simulateur,coord_depart,coord_arrivee):

        self.sim=simulateur
        self.dep=coord_depart   #liste des coord de départ (lat,lon)
        self.arr=coord_arrivee  #liste des coord d'arrivée (lat,lon)
        noeuddep=Node(0,coord_depart[0],coord_depart[1])
        print(noeuddep.time)
        self.openlist=[noeuddep]
        self.closelist=[]
        self.currentnode = Node(0,coord_depart[0],coord_depart[1])
        self.liste_actions = []
        
    def reset(self,coord_depart=None,coord_arrivee=None):
        if not coord_depart==None:
            self.dep=coord_depart   #liste des coord de départ (lat,lon)
        if not coord_arrivee==None:
            self.arr=coord_arrivee  #liste des coord d'arrivée (lat,lon)
        
        noeuddep=Node(0,self.dep[0],self.dep[1])
        self.openlist=[noeuddep]
        self.closelist=[]
        self.currentnode=None
        self.liste_actions = []
        return None
        
    def petitfopen(self):
        """retourne le noeud de plus petit f se trouvant dans openlist"""
        petit=self.openlist[0]
        for node in self.openlist:
            if petit.f>node.f:
                petit=node
        return petit
        
    def ajout_openlist_trie_par_f_et_g(self,noeud_voisin):
        """permet de ne pas avoir à chercher le plus petit f de l'openlist"""
        f = noeud_voisin.f
        g = noeud_voisin.g
        i = 0
        search = True
        while search:
            print(i)
            if f == self.openlist[i].f:
                if g > self.openlist[i].g:
                    self.openlist.insert(i,noeud_voisin)
                    search = False
            elif f < self.openlist[i].f:
                self.openlist.insert(i,noeud_voisin)
                search = False
            i+=1
            if i == len(self.openlist)-1:
                search = False
        return None
        
    def currentvoisin(self):
        """retourne la liste des noeuds voisins du noeud courant"""
        L=[]
        A = [] #liste de actions réalisable à partir de l'état courant soit 6 caps
        nb_caps_possible = 6
        pas_en_degre = int(360/nb_caps_possible)
        Dt=(self.sim.times[self.currentnode.time+1]-self.sim.times[self.currentnode.time])
        Vmax_iboat = 6 #metre par seconde
        seconde_jour = 86400
        
        for i in range(nb_caps_possible):
            A.append(i*pas_en_degre)
        
        current_state = [self.currentnode.time,self.currentnode.lat,self.currentnode.lon]

        for a in A:
            self.sim.reset(current_state)
            etat_voisin = self.sim.doStep(a)
            g = self.currentnode.g + Dt
            dist,bear = self.sim.getDistAndBearing(etat_voisin[1:3],self.arr)
            h = dist/Vmax_iboat/seconde_jour
            f = g+h
            voisin = Node(etat_voisin[0],etat_voisin[1],etat_voisin[2],self.currentnode,a,g,h,f)
            L.append(voisin)
        return L   
    
    def testpresclose(self,node):
        """teste si le node se trouve dans la closelist"""
        state=0
        meme=None
        for traite in self.closelist:
            if traite.lat==node.lat and traite.lon==node.lon: #doit-on mettre une fourchette de distance?
                state=1
                meme=traite
                break
        if state:
            return True,meme
        else:
            return False,meme
    
    def testpresopen(self,node):
        """teste si le node se trouve dans la openlist"""
        state=0
        meme=None
        for traite in self.openlist:
            if traite.lat==node.lat and traite.lon==node.lon: #doit-on mettre une fourchette de distance?
                state=1
                meme=traite
                break
        if state:
            return True,meme
        else:
            return False,meme
    
    def solver(self):
        """retourne une liste de caps solution du pb de navigation en optimisant le temps de parcours"""
        state=0 # état nul tant que le noeud d'arrivée n'est pas ajouté à la closelist
        while len(self.openlist)!=0 and state==0:
            print(self.currentnode)
            self.currentnode=self.petitfopen()
            self.closelist.append(self.currentnode)
            self.openlist.remove(self.currentnode)
            try:
                etre_arrive = Tree.isStateAtDest(self.arr, self.currentnode.pere.give_state(),self.currentnode.give_state())
            except:
                etre_arrive= [False, 0]
                
            if (self.currentnode.lat==self.arr[0] and self.currentnode.lon==self.arr[1]) or etre_arrive[0]:
                state=1
                nodearr=self.currentnode
                break
            adjacent=self.currentvoisin()
            for voisin in adjacent:
                cond_close,meme_close=self.testpresclose(voisin)
                cond_open,meme_open=self.testpresopen(voisin)
                if not (cond_close or cond_open):
                    self.openlist.append(voisin)
                else:
                    if cond_close:
                        if meme_close.g > voisin.g:
                            self.openlist.remove(meme_close)
                            self.openlist.append(voisin)
                    else:
                        if meme_open.g > voisin.g:
                            self.openlist.remove(meme_open)
                            self.openlist.append(voisin)
        if not state:
            return []
        else:
            listecapsarr=[]
            while nodearr.pere is not None:
                listecapsarr.append(nodearr.act)
                nodearr=nodearr.pere
            self.liste_actions = listecapsarr[::-1]
            return self.liste_actions
    
    def solverplus(self):
        """retourne une liste de caps solution du pb de navigation en optimisant le temps de parcours"""
        state=0 # état nul tant que le noeud d'arrivée n'est pas ajouté à la closelist
        while len(self.openlist)!=0 and state==0:
            self.currentnode=self.openlist[0]
            self.closelist.append(self.currentnode)
            self.openlist.remove(self.currentnode)
            etre_arrive = Tree.isStateAtDest(self.arr, self.currentnode.pere.give_state(),self.currentnode.give_state())
            if (self.currentnode.lat==self.arr[0] and self.currentnode.lon==self.arr[1]) or etre_arrive[0]:
                state=1
                nodearr=self.currentnode
                break
            adjacent=self.currentvoisin()
            for voisin in adjacent:
                cond_close,meme_close=self.testpresclose(voisin)
                cond_open,meme_open=self.testpresopen(voisin)
                if not (cond_close or cond_open):
                    self.ajout_openlist_trie_par_f_et_g(voisin)
                else:
                    if cond_close:
                        if meme_close.g > voisin.g:
                            self.openlist.remove(meme_close)
                            self.ajout_openlist_trie_par_f_et_g(voisin)
                    else:
                        if meme_open.g > voisin.g:
                            self.openlist.remove(meme_open)
                            self.ajout_openlist_trie_par_f_et_g(voisin)
        if not state:
            return []
        else:
            listecapsarr=[]
            while nodearr.pere is not None:
                listecapsarr.append(nodearr.act)
                nodearr=nodearr.pere
            self.liste_actions = listecapsarr[::-1]
            return self.liste_actions