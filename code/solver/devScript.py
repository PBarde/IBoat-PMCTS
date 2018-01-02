#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:53:03 2018

@author: paul
"""
import sys

sys.path.append("../model/")
import myTree as mt

node=mt.Node()

for hist in node.Values : 
  print(hist)