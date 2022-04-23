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

#obj=ivae.IVAE(reconst_coef=1,kl_coef=1,classifier_coef=0.1)
obj1=ivae.IVAE(reconst_coef=100000,kl_coef=0.0001*512,classifier_coef=1000)
obj2=ivae.IVAE(reconst_coef=100000,kl_coef=0.0001*512,classifier_coef=0)

save_address1="result_files/cls_1000_date_12_27"
save_address2="result_files/cls_0_date_12_27"

def run(obj,save_address):
    ##########
    if model_init:
        obj.model_initialiaze()
    ##########
    if model_tobe_trained:
        lr=1e-2
        print(lr)
        obj.model_training(epochs=100,learning_rate=lr)
        
        lr=5e-3
        print(lr)
        obj.model_training(epochs=150,learning_rate=lr)
        
        lr=2e-3
        print(lr)
        obj.model_training(epochs=150,learning_rate=lr)
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
    #obj.model_save(address=save_address+".pt")
    #obj.save_residuals(address=save_address+'_residuals.pkl')
    ##########
#run(obj1,save_address1)
#run(obj2,save_address2)


####################
obj1=obj1
save_address=save_address1
obj1.model_load(address=save_address+".pt")
obj1.load_residuals(address=save_address+'_residuals.pkl')
obj1.generate_test_results()

for i in range(10): 
    line_decoded = obj1.traverse(number_of_images=21, start_id=2, end_id=10, model_name="supervised_")
####################

####################
obj2=obj2
save_address=save_address2
obj2.model_load(address=save_address+".pt")
obj2.load_residuals(address=save_address+'_residuals.pkl')
obj2.generate_test_results()
for i in range(10): 
    line_decoded = obj2.traverse(number_class=i, number_of_images=21, start_id=2, end_id=10,model_name="UNsupervised_")
####################

obj=obj1
save_address=save_address1
obj.model_load(address=save_address+".pt")
obj.load_residuals(address=save_address+'_residuals.pkl')
obj.generate_test_results()
obj.display_images_real_vs_synthetic(number_class=5,image_number=32,image_shape=28)
obj.test_BCE_tracker[-1]
np.mean(obj.test_BCE_tracker[770:801])
obj.regression_analysis(obj.zs,obj.y_last)
obj.plot_residuals(init_index=50)


obj=obj2
save_address=save_address2
obj.model_load(address=save_address+".pt")
obj.load_residuals(address=save_address+'_residuals.pkl')
obj.generate_test_results()
obj.display_images_real_vs_synthetic(number_class=5,image_number=32,image_shape=28)
np.mean(obj.test_BCE_tracker[770:801])
obj.regression_analysis(obj.zs,obj.y_last)
obj.plot_residuals(init_index=90)

import matplotlib.pyplot as plt


#obj.regression_analysis(obj.means,obj.labels)
tsne_mat1,umap_mat1,pca_mat1,Y1=obj1.calculate_lower_dimensions(obj1.zs,obj1.y_last,N=10000)
tsne_mat2,umap_mat2,pca_mat2,Y2=obj2.calculate_lower_dimensions(obj2.zs,obj2.y_last,N=10000)


obj=obj1
tsne_mat=tsne_mat1
umap_mat=umap_mat1
pca_mat=pca_mat1
Y=Y1


obj=obj2
tsne_mat=tsne_mat2
umap_mat=umap_mat2
pca_mat=pca_mat2
Y=Y2



obj.plot_lower_dimension(tsne_mat,Y,projection='3d',size_dot=20)
obj.plot_lower_dimension(tsne_mat,Y,projection='2d',size_dot=20)
obj.plot_lower_dimension(umap_mat,Y,projection='3d',size_dot=30)
obj.plot_lower_dimension(umap_mat,Y,projection='2d',size_dot=30)
obj.plot_lower_dimension(pca_mat,Y,projection='3d',size_dot=3)
obj.plot_lower_dimension(pca_mat,Y,projection='2d',size_dot=3)


obj.display_images_real_vs_synthetic(number_class=4,image_number=32,image_shape=28)

line_decoded = obj.traverse(number_class=5, number_of_images=21, start_id=0, end_id=60)
#obj.pipeline(epochs=10)
#obj.traverse_multiple(number_class=3,number_of_images=20,start_id=10,end_id=60,file_path_root="multiple_gif")
number_class = 2
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




