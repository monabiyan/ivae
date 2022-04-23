#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:21:03 2022

@author: mnabian
"""

import pandas as pd
#############################
root_file="/Users/mnabian/Desktop/"
#############################
genes=pd.read_csv(root_file + "clone_cancer_sc/E-MTAB-8410-normalised-files/E-MTAB-8410.aggregated_filtered_normalised_counts.mtx_rows",sep='\t',header=None)
genes.columns
genes.loc[:,0].equals(genes.loc[:,1])
genes = genes.loc[:,0]
genes = list(genes.values)
#############################
cells = pd.read_csv(root_file + "clone_cancer_sc/E-MTAB-8410-normalised-files/E-MTAB-8410.aggregated_filtered_normalised_counts.mtx_cols",sep='\t',header=None)
cells=list(cells.loc[:,0].values)
#############################
from scipy.io import mmread
sc = mmread("/Users/mnabian/Desktop/clone_cancer_sc/E-MTAB-8410-normalised-files/E-MTAB-8410.aggregated_filtered_normalised_counts.mtx")
sc = sc.toarray()
sc=sc.T
sc.shape
len(cells)
len(genes)
sc_df=pd.DataFrame(sc)
sc_df.shape
sc_df.columns=genes
sc_df['cells']=cells
#############################
ff=pd.read_csv(root_file +"/clone_cancer_sc/E-MTAB-8410.clusters.tsv",sep='\t')
print(ff.nunique())
#############################
ant=pd.read_csv(root_file +"clone_cancer_sc/ExpDesign-E-MTAB-8410.tsv",sep='\t')
ant.columns
ant=ant.rename(columns={'Sample Characteristic[sampling site]':'site'})
ant=ant.rename(columns={'Sample Characteristic[organism part]':'organism_part'})
ant=ant.rename(columns={'Sample Characteristic[sex]':'sex'})
ant=ant.rename(columns={'Factor Value[inferred cell type - ontology labels]':'celltype_ontology'})
ant=ant.rename(columns={'Factor Value[inferred cell type - authors labels]':'celltype_authors'})
ant=ant.rename(columns={'Assay':'cells'})
ant = ant.loc[:,['cells','site','organism_part','sex','celltype_ontology','celltype_authors']]
#############################
zero_rows = (sc_df == 0).astype(int).sum(axis=1)
zero_columns = (sc_df == 0).astype(int).sum(axis=0)

sum(list(zero_rows>23900))
sum(list(zero_columns<60000))

import numpy as np
ant['celltype_ontology']!=np.nan


m=sc_df.isin([0]).sum(axis=1)

sc_df_filtered=sc_df.loc[ant['celltype_ontology'].notna(),list(zero_columns<60000)]
sc_df_filtered.shape
sum(sc_df_filtered.var()>24000)


sc_df_filtered2=sc_df_filtered.loc[:,list(sc_df_filtered.var()>24000)+[True]]
sc_df_filtered2.to_csv("./sc_counts_filtered.csv")
ant.to_csv("./sc_annotations.csv")
#############################
sc_df_filtered2 = pd.read_csv("./sc_counts_filtered.csv")
ant = pd.read_csv("./sc_annotations.csv")

df=pd.DataFrame({'cells':sc_df_filtered2['cells'].tolist()})
ant=pd.merge(df,ant,how='left',on='cells')

ant.shape
sc_df_filtered2.shape
ant.columns

df=pd.DataFrame({'celltype_ontology':list(ant['celltype_ontology'].unique()),
                 'celltype_id':list(range(0,29))})

ant=pd.merge(left=ant,right=df,how='left',on='celltype_ontology')
ant.to_csv("./sc_annotations_final.csv",index=False)
#############################
sc_df_filtered2['cells']=ant['celltype_id']
sc_df_filtered2=sc_df_filtered2.rename(columns={'cells':'Y'})
sc_df_filtered2.to_csv("./sc_counts_filtered_final.csv",index=False)
#############################
#############################
ant=pd.read_csv("./sc_annotations_final.csv")
sc_df_filtered2=pd.read_csv("./sc_counts_filtered_final.csv")

############