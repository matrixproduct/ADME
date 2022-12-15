"""
Train and test various model on ADME datasets.
Return performance for each dataset as DataFrame.
"""
import numpy as np
import pandas as pd
import pickle
import time



from tdc.single_pred import ADME
from tdc.utils import retrieve_dataset_names
from tdc.benchmark_group import admet_group

from models.model_gcn import gcn_predict
from models.model_transformer import transformer_predict
from models.model_tree import tree_benchmark_predict

import utils
from utils import classification_metrics, regression_metrics

from config import best_params



def benchmark_predict(group, benchmark, name, model_predict, **kwargs):
    '''Wrapper for model_predict(). Converts benchmark into train, valid and test split.'''
    seed = 566
    train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    split = {'train': train, 'valid': valid, 'test': benchmark['test']}
    valid_outputs, test_outputs = model_predict(split, **kwargs)
    return test_outputs
    
start = time.time()

#%% Data load
group = admet_group(path='data/')
adme_dataset_names = retrieve_dataset_names('ADME')

#%% Loop over datasets and models

# Specify dataset names
set_names = adme_dataset_names 
# set_names = ['vdss_lombardo','half_life_obach']

# models to be trained and tested
models = {'gcn': gcn_predict, 'transformer': transformer_predict,
           'tree': tree_benchmark_predict}

set_performance = {}  # test performance for all models and datasets

for set_name in set_names:  
    if set_name in best_params:        
        benchmark = group.get(set_name) 
        # determine type of the task and number of classes
        classification, num_classes =  utils.get_problem_type(benchmark)
        print(set_name, classification, num_classes)
        
        kwargs = {'set_name':set_name}
        pred_df = benchmark['test'][['Y']]
        performances = []
        for model_name, model_predict in models.items():
            print(f'\n{model_name}\n')
            # train and test model 
            if  model_name == 'tree':
                test_outputs = model_predict(benchmark, **kwargs)
            else:
                test_outputs = benchmark_predict(group, benchmark, set_name, model_predict, **kwargs)
            
            # calculate performance    
            if classification:
                metrics = classification_metrics(pred_df['Y'], test_outputs)
            else:
                metrics = regression_metrics(pred_df['Y'], test_outputs)
            df_performance = pd.DataFrame(metrics, index=[model_name])  
            performances.append(df_performance)
        df_performances = pd.concat(performances) 
        # save performnce 
        df_performances.to_csv(f'./results/{set_name}.csv')    
        set_performance[set_name] = df_performances

end = time.time()

print(f'Execution time: {(end-start) / 60:.02f} minutes')

#%% Save models' performance into pickle file
filename = 'performance.pickle'
with open(filename, 'wb') as handle:
    pickle.dump(set_performance, handle, protocol=-1)    

