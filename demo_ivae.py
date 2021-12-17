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
model_file_address='./test_model.pt'

obj=ivae.IVAE()

if model_init:
    obj.model_initialiaze()
if model_tobe_trained:
    obj.model_training(epochs=20,learning_rate=1e-4)
obj.model_save(address=model_file_address)
obj.model_load(address=model_file_address)
obj.plot_residuals()

obj.generate_test_results()

#obj.regression_analysis(obj.means,obj.labels)

tsne_mat,umap_mat,Y=obj.calculate_lower_dimensions(obj.miu_last,obj.y_last,N=1000)
obj.plot_lower_dimension(tsne_mat,Y,projection='3d')
obj.plot_lower_dimension(tsne_mat,Y,projection='2d')
obj.plot_lower_dimension(umap_mat,Y,projection='3d')
obj.plot_lower_dimension(umap_mat,Y,projection='2d')

obj.display_images_real_vs_synthetic()

line_decoded = obj.traverse(number_class=3, number_of_images=20, start_id=10, end_id=60)
#obj.pipeline(epochs=10)
#obj.traverse_multiple(number_class=3,number_of_images=20,start_id=10,end_id=60,file_path_root="multiple_gif")
number_class = 3
number_of_additional_data=4000
synthetic_physical_data = obj.generate_synthetic_data(obj.model,obj.miu_last,obj.y_last,number_class=number_class,number_of_additional_data=number_of_additional_data)
synthetic_physical_data.shape

########## (optional: Saving GIF from Augmented images)
synthetic_augmented_images = np.fliplr(np.flip((((synthetic_physical_data.reshape(number_of_additional_data,28,28)*256)))))
#obj.save_GIF(synthetic_augmented_images,file_path_root="synthetic_augmented_images",indicator=str(number_class),speed=7)
##########
obj.append_augmented_data_to_original(synthetic_physical_data,number_class=number_class,number_of_additional_data=number_of_additional_data)
# Original data test
print(obj.x_last.cpu().numpy().shape)
print(obj.y_last.cpu().numpy().shape)
# Original data + augmented data
print(obj.original_with_augmented_data_all_X.cpu().numpy().shape)
print(obj.original_with_augmented_data_all_lables.cpu().numpy().shape)



tsne_mat_augmented,umap_mat_augmented,Y_augmented = obj.calculate_lower_dimensions(obj.original_with_augmented_data_all_X,obj.original_with_augmented_data_all_lables,N=1000)

obj.plot_lower_dimension(tsne_mat_augmented,Y_augmented,projection='2d')
obj.plot_lower_dimension(tsne_mat_augmented,Y_augmented,projection='3d')
obj.plot_lower_dimension(umap_mat_augmented,Y_augmented,projection='2d')
obj.plot_lower_dimension(umap_mat_augmented,Y_augmented,projection='3d')




