import sys
sys.path.insert(1, '/Users/mnabian/Documents/GitHub/ivae')
import ivae 
import pandas as pd
import numpy as np
##############################################################
'''
ant = pd.read_csv("/Users/mnabian/Documents/GitHub/ivae/sc_annotations_final.csv")
for c in ant.columns.tolist():
    print(c)
ant['site'].unique()
ant['site_id']=ant['site']
ant.loc[ant['site_id']=='normal tissue adjacent to neoplasm',['site_id']]=0
ant.loc[ant['site_id']=='tumour border',['site_id']]=5
ant.loc[ant['site_id']=='tumour core',['site_id']]=10
sc_df_filtered2=pd.read_csv("/Users/mnabian/Documents/GitHub/sc_data/sc_counts_filtered_final.csv")
sc_df_filtered2['YY']=ant['site_id']
sc_df_filtered2.to_csv("/Users/mnabian/Documents/GitHub/sc_data/sc_counts_filtered_final.csv")
'''

sc_df_filtered2=pd.read_csv("/Users/mnabian/Documents/GitHub/sc_data/sc_counts_filtered_final.csv")

for c in sc_df_filtered2.columns.tolist():
    print(c)
    break
#ant=ant.loc[:,['cells','site']]

##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
def gene_id_to_symbol(genes):
    import mygene
    mg = mygene.MyGeneInfo()
    geneSyms = mg.querymany(genes , scopes='ensembl.gene', fields='symbol', species='human')
    return(geneSyms)

def gene_symbol_to_id(genes):
    import mygene
    mg = mygene.MyGeneInfo()
    geneSyms = mg.querymany(genes , scopes='symbol', fields='ensembl.gene', species='human')
    return(geneSyms)

gene_symbol_to_id(['CCDC83'])
gene_id_to_symbol(['ENSG00000150676'])
gene_symbol_to_id(['COL15A1'])
##############################################################
##############################################################   
df_XY=sc_df_filtered2
labels1=df_XY['Y'].tolist()
labels2=df_XY['YY'].tolist()
df_XY=df_XY/100000
df_XY['Y']=labels1
df_XY=df_XY.drop(columns=['YY'])
df_XY.shape
df_XY.head()
##############################################################   
##############################################################
model_init=True
model_tobe_trained=True
model_file_address='./test_model.pt'
#sc_df_filtered2=sc_df_filtered2.drop(columns=['Unnamed: 0','Y','Y','Y'])


obj1=ivae.IVAE(df_XY=df_XY,reconst_coef=1000000,kl_coef=0.0001*512,classifier_coef=10,test_ratio=1)

save_address1="clone_sc1"
def run(obj,save_address):
    ##########
    if model_init:
        obj.model_initialiaze()
    ##########
    if model_tobe_trained:
        lr=1e-2
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        lr=5e-3
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        lr=2e-3
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        obj.model_save(address=save_address+".pt")
        obj.save_residuals(address=save_address+'_residuals.pkl')
        lr=1e-3
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        lr=5e-4
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        obj.model_save(address=save_address+".pt")
        obj.save_residuals(address=save_address+'_residuals.pkl')
        
        lr=1e-5
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)
        
        lr=5e-6
        print(lr)
        obj.model_training(epochs=50,learning_rate=lr)

                
    ##########
#run(obj1,save_address1)

#obj1.model_save(address=save_address1+".pt")
#obj1.save_residuals(address=save_address1+'_residuals.pkl')




obj1=obj1
save_address=save_address1
obj1.model_load(address=save_address+".pt")
obj1.load_residuals(address=save_address+'_residuals.pkl')
obj1.generate_test_results()
obj1.plot_residuals(init_index=110)
obj1.regression_analysis(obj1.zs,obj1.y_last)




tsne_mat,umap_mat,pca_mat,Y=obj1.calculate_lower_dimensions(obj1.zs,obj1.y_last,N=2000)
obj1.plot_lower_dimension(tsne_mat,Y,projection='3d')
obj1.plot_lower_dimension(tsne_mat,Y,projection='2d')
obj1.plot_lower_dimension(umap_mat,Y,projection='3d')
obj1.plot_lower_dimension(umap_mat,Y,projection='2d')
obj1.plot_lower_dimension(pca_mat,Y,projection='3d')
obj1.plot_lower_dimension(pca_mat,Y,projection='2d')


ant.loc[ant['site']=='normal tissue adjacent to neoplasm'].loc[ant['celltype_id']==3]['index_no']
ant.loc[ant['site']=='tumour core'].loc[ant['celltype_id']==3]['index_no']


ant_df=pd.DataFrame({'Y':labels1,'YY':labels2,'index':list(range(0, len(labels1)))})
def mean_traversal(cell_type_id):
    import random
    healthy = list(ant_df.loc[ant_df['YY']==0].loc[ant_df['Y']==cell_type_id]['index'])
    cancer = list(ant_df.loc[ant_df['YY']==10].loc[ant_df['Y']==cell_type_id]['index'])
    #healthy = [i for i, x in enumerate(YY) if x == 0]
    #cancer = [i for i, x in enumerate(YY) if x == 10]
    print(len(healthy),len(cancer))
    h_max=min(50,len(healthy))
    c_max=min(50,len(cancer))
    traversal_step=100
    line_decoded=np.zeros(shape=(traversal_step, 2083,h_max*c_max))
    index=0
    
    for h in random.sample(healthy, h_max):
        for c in random.sample(cancer, c_max):
            #print((sc_df_filtered2.iloc[c,:-1]-sc_df_filtered2.iloc[h,:-1]).mean())
            ss =obj1.traverse(number_of_images=traversal_step, start_id=h, end_id=c,model_name="supervised_")
            #print(ss.shape)
            #ss = ss/ss[0]
            #print(ss)
            #line_decoded = np.add(line_decoded,ss)
            line_decoded[:,:,index]=ss
            index=index+1
    #line_decoded = line_decoded/(50*50)
    line_decoded_med = np.median(line_decoded,axis=2)
    line_decoded_mean = np.mean(line_decoded,axis=2)
    line_decoded_std = np.std(line_decoded,axis=2)
    print(line_decoded_med)
    
    gg_med=pd.DataFrame(line_decoded_med)
    gg_med.columns=df_XY.columns[0:-1]
    
    gg_mean=pd.DataFrame(line_decoded_mean)
    gg_mean.columns=df_XY.columns[0:-1]
    
    gg_std=pd.DataFrame(line_decoded_std)
    gg_std.columns=df_XY.columns[0:-1]
    
    #gg= gg.div(gg.iloc[0])
    return(gg_med,gg_mean,gg_std)
    #return(line_decoded)
    
ff=dict()
ff['mean']=dict()
ff['med']=dict()
ff['std']=dict()

for i in range(len(set(ant_df['Y']))):
    print(i)
    ff['mean'][str(i)],ff['med'][str(i)],ff['std'][str(i)]=mean_traversal(i)

import pickle
with open('results_dict.pkl', 'wb') as f:
    pickle.dump(ff, f)
        
with open('results_dict.pkl', 'rb') as f:
    ff = pickle.load(f)





def find_celltype_from_id(k):
    return(list(ant.loc[ant['celltype_id']==k,['celltype_ontology']]['celltype_ontology'])[0])

def plt_gene(mm,g_symbol,gene_id):
    
    plt.xlabel("healthy to cancer")
    plt.legend()
    
import matplotlib.pyplot as plt

down_genes=[""]



intertested_genes=["COL15A1","RHOB","HSPG2","SERPINE1","COL4A2","LAMA4"
                   ,"COL4A1","SPARC","PFN1","LGALS1","S100A6","KLF6","HTRA1"]


#######################################
for g in intertested_genes:
    print(g)
    g_id=gene_symbol_to_id(g)[0]['ensembl']['gene']
    plt.plot(ff_med[g_id],label=g)    
    plt.xlabel("healthy to cancer")
    plt.legend()
#######################################    
    
    
#plt.ylim(0.8,1.2)
plt.title('celltype_3')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
    
ff_med,ff_mean,ff_std = mean_traversal(k) 
def plot_gene_expression_mean_std(gene_name_list,celltype_id):

    for g in gene_name_list:
        g_id=gene_symbol_to_id(g)[0]['ensembl']['gene']       
        #plt.plot(ff_med[g_id],label=g)
        y = np.array(ff_mean[g_id])*1000000
        y_med=np.array(ff_med[g_id])*1000000
        x = np.arange(0,len(list(y))) # Effectively y = x**2
        e = np.array(ff_std[g_id])*1000000
        print(e)
        plt.errorbar(x,y,e, linestyle='None', marker='.')
        plt.errorbar(x,y_med, linestyle='None', marker='.')
        #plt.ylim(0,3)
        plt.title(find_celltype_from_id(k)+" : "+g)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
    

for i in range(2):
    plot_gene_expression_mean_std(intertested_genes,i)
for i in range(1,22):
    plot_gene_expression_mean_std(intertested_genes,1)

########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################

plt_gene("COL15A1","ENSG00000204291")   
plt_gene("COL4A2","ENSG00000134871")   
plt_gene("COL4A1","ENSG00000187498")   
plt_gene("CD74","ENSG00000019582")   
plt_gene("CD320","ENSG00000167775")    

COL15A1="ENSG00000204291"
plt.plot(mm[COL15A1])


COL4A2 = "ENSG00000134871"
plt.plot(mm[COL4A2])

COL4A1="ENSG00000187498"
plt.plot(mm[COL4A1])

CD74="ENSG00000019582"
plt.plot(mm[CD74])

CD320 = "ENSG00000167775"
plt.plot(mm[CD320])











line_decoded = obj1.traverse(number_of_images=100, start_id=3911, end_id=3,model_name="supervised_")
line_decoded=line_decoded*1000000
line_decoded.shape
mm=pd.DataFrame(line_decoded)
mm.columns=sc_df_filtered2.columns[0:-1]
import seaborn as sns
sns.heatmap(mm)

mm= mm.div(mm.iloc[0])

import matplotlib.pyplot as plt

COL15A1="ENSG00000204291"
plt.plot(mm[COL15A1])

COL4A2 = "ENSG00000134871"
plt.plot(mm[COL4A2])

COL4A1="ENSG00000187498"
plt.plot(mm[COL4A1])

CD74="ENSG00000019582"
plt.plot(mm[CD74])

CD320 = "ENSG00000167775"
plt.plot(mm[CD320])



for i in range(20):
    plt.plot(line_decoded[:,i])
    plt.show()

for i in range(10):
    for j in range(10):
        plt.plot(line_decoded[:,i],line_decoded[:,j])
        plt.show()

        
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
number_class = 4
number_of_additional_data=40
synthetic_physical_data = obj1.generate_synthetic_data(obj1.model,obj1.miu_last,obj1.y_last,number_class=number_class,number_of_additional_data=number_of_additional_data)
synthetic_physical_data.shape


obj1.append_augmented_data_to_original(synthetic_physical_data,number_class=number_class,number_of_additional_data=number_of_additional_data)

# Original data test
print(obj1.x_last.cpu().numpy().shape)
print(obj1.y_last.cpu().numpy().shape)
# Original data + augmented data
print(obj1.original_with_augmented_data_all_X.cpu().numpy().shape)
print(obj1.original_with_augmented_data_all_lables.cpu().numpy().shape)


tsne_mat_augmented,umap_mat_augmented,Y_augmented = obj1.calculate_lower_dimensions(obj1.original_with_augmented_data_all_X,obj1.original_with_augmented_data_all_lables,N=1000)
obj1.plot_lower_dimension(tsne_mat_augmented,Y_augmented,projection='2d')
obj1.plot_lower_dimension(tsne_mat_augmented,Y_augmented,projection='3d')
obj1.plot_lower_dimension(umap_mat_augmented,Y_augmented,projection='2d')
obj1.plot_lower_dimension(umap_mat_augmented,Y_augmented,projection='3d')




ant['site'].unique()



ant.loc[ant['site']=='tumour core'].loc[ant['celltype_id']==1]['cells']
ant.shape
ant['index_no']=list(range(0,25480))
ant.loc[ant['cells']=='SAMEA6057390-AAACCTGAGAAACCTA']['index_no'].values[0]
ant.loc[ant['cells']=='SAMEA6057388-AAACCTGTCACCACCT']['index_no'].values[0]
ant.loc[ant['celltype_id']==1]['celltype_ontology'].unique()








