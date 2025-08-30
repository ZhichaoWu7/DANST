from collections import defaultdict

option_list = defaultdict(list)


def get_option_list(dataset):
    if dataset == 'MBC':
        # Input parameters
        option_list['data_dir'] = '/home/featurize/work/DANST_model/DANST/data/Mouse_Brain_Coronal/'
        option_list['ref_dataset_name'] = 'scRNA.h5ad' #scRNA_data
        option_list['target_dataset_name'] = 'ST.h5ad' #ST_data
        option_list['random_type']= 'annotation_1' #the celltype label in scRNA_data
        option_list['type_list'] = None #the selected celltypes
         # data preprocessing                 
        option_list['sample_list']= True # if have simulated ST spot，set True，otherwise False
        option_list['ref_sample_num'] = 8000 #the count of simulated ST spots
        option_list['sample_size'] = 10 #the count of cells in one simulated ST spot
        option_list['HVP_num'] = 5000
        option_list['DVG_num'] = 30  # DVGs        
        option_list['down_sample'] =  True #if do downsample in simulated ST data
        option_list['get_location'] = {'n_clusters':25,'start':0.1, 'end':4.0,'increment' : 0.01} # clustering parameters
        option_list['n_neighbors']= 10 
        option_list['target_type'] = "real"
        option_list['target_sample_num'] = 1000
         # Training parameters 
        option_list['batch_size' ] =500
        option_list['AE_epochs'] = 200
        option_list['AE_learning_rate'] = 0.001
        option_list['DANN_epochs'] = 200
        option_list['DANN_learning_rate1'] = 0.005
        option_list[ 'DANN_learning_rate2'] = 0.005  
            
        # Output parameters
        option_list['SaveResultsDir'] = "/home/featurize/work/DANST_model/DANST/result/MBC/"

    return option_list
