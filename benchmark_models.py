import numpy as np
import pandas as pd



from tdc.single_pred import ADME
from tdc.utils import retrieve_dataset_names
from tdc.benchmark_group import admet_group

from models.model_gcn import gcn_predict
from models.model_transformer import transformer_predict
from models.model_tree import tree_benchmark_predict

import utils
from utils import classification_metrics, regression_metrics



def benchmark_predict(group, benchmark, name, model_predict, **kwargs):
    seed = 566
    train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
    split = {'train': train, 'valid': valid, 'test': benchmark['test']}
    valid_outputs, test_outputs = model_predict(split, **kwargs)
    return test_outputs
    

#%% Data load
group = admet_group(path = 'data/')
adme_dataset_names = retrieve_dataset_names('ADME')

# adme_set_name = 'Caco2_Wang'
# adme_set_name = 'Bioavailability_Ma'

# set_names = adme_dataset_names
# set_names  = ['Caco2_Wang', 'Bioavailability_Ma']
set_names  = ['Bioavailability_Ma']
# set_names  = ['Caco2_Wang']


models = {'gcn': gcn_predict, 'transformer': transformer_predict,
           'tree': tree_benchmark_predict}
# models = {'gcn': gcn_predict} 
# models = {'transformer': transformer_predict}
# models =  {'gcn': gcn_predict,'tree': tree_benchmark_predict}

set_performance = {}

for set_name in set_names:
    # adme_data = ADME(name=set_name)
    # split = adme_data.get_split()
    benchmark = group.get(set_name) 
    classification, num_classes =  utils.get_problem_type(benchmark)
    print(set_name, classification, num_classes)
    
    kwargs = {'set_name':set_name}
    pred_df = benchmark['test'][['Y']]
    performances = []
    for model_name, model_predict in models.items():
        print(f'\n{model_name}\n')
        # valid_outputs, test_outputs = model_predict(split, **kwargs)
        if  model_name == 'tree':
            test_outputs = model_predict(benchmark, **kwargs)
        else:
            test_outputs = benchmark_predict(group, benchmark, set_name, model_predict, **kwargs)
        # print(test_outputs.shape)
        if classification:
            metrics = classification_metrics(pred_df['Y'], test_outputs)
        else:
            metrics = regression_metrics(pred_df['Y'], test_outputs)
        df_performance = pd.DataFrame(metrics, index=[model_name])  
        performances.append(df_performance)
    set_performance[set_name] = pd.concat(performances)    
