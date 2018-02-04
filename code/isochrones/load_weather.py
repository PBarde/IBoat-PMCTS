#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:45:58 2018

@author: jean-mi
"""

#télécharger fichier du 11/01/2018 à 13h Z

import sys
sys.path.append("../solver")
import forest

forest.Forest.download_scenarios('20180111')