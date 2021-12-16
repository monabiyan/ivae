#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 01:33:35 2021

@author: mnabian
"""

##############################################################
##############################################################
import sys
sys.path.insert(1, '/Users/mnabian/Documents/GitHub/ivae')
import ivae 
import pandas as pd
##############################################################
##############################################################
obj=ivae.IVAE()
obj.pipeline(epochs=10)
