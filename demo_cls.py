#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:00:52 2021

@author: mnabian
"""
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
import numpy as np
##############################################################
##############################################################
model_init=True
model_tobe_trained=True
model_file_address='./test_classification_model.pt'

obj=ivae.CLS()
save_address="classification_date_12_29"

def run(obj,save_address):
    ##########
    if model_init:
        obj.model_initialiaze()
    ##########
    if model_tobe_trained:
        lr=1e-2
        print(lr)
        obj.model_training(epochs=120,learning_rate=lr)
        
        lr=5e-3
        print(lr)
        obj.model_training(epochs=120,learning_rate=lr)
        
        lr=2e-3
        print(lr)
        obj.model_training(epochs=120,learning_rate=lr)
        #obj.model_save(address=save_address+".pt")
        #obj.save_residuals(address=save_address+'_residuals.pkl')
        
        
        
        lr=1e-3
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        
        lr=5e-4
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        
        lr=1e-4
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        #obj.model_save(address=save_address+".pt")
        #obj.save_residuals(address=save_address+'_residuals.pkl')
        
        lr=1e-5
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        
        lr=5e-6
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        
    ##########
    obj.model_save(address=save_address+".pt")
    obj.save_residuals(address=save_address+'_residuals.pkl')
    ##########
run(obj,save_address)
obj.plot_residuals(init_index=0)

obj.model_load(address=save_address+".pt")
obj.load_residuals(address=save_address+'_residuals.pkl')

mm=obj.read_gifs(filename="multiple_gifmultiple_supervised_2_10_83.gif", asNumpy=True)
mm=np.array(mm)
mm.shape



