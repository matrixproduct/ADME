"""
We used the approach developed in 
'Accurate ADMET Prediction with XGBoost' by Hao Tian, Rajas Ketkar, Peng Tao
https://arxiv.org/abs/2204.07532v2

https://github.com/smu-tao-group/ADMET_XGBoost
"""

import numpy as np
import xgboost
import utils
from config import best_params as saved_best_params



tree_method = 'gpu_hist' if utils.gpu_available else 'hist'

def tree_benchmark_predict(benchmark, **kwargs):
    '''
    Train and test the tree model on one of the ADME datasets.
    Load pre-calculated features from files.  
    '''
               
    classification, num_classes =  utils.get_problem_type(benchmark)
    name = kwargs.get('set_name').lower()   
    best_params = saved_best_params[name]
    
    
    xgb = xgboost.XGBClassifier(
        #tree_method='gpu_hist',
        tree_method=tree_method,
        **best_params,
        random_state=566) if classification else\
        xgboost.XGBRegressor(
        #tree_method='gpu_hist',
        tree_method='hist',
        **best_params,
        random_state=566)    
    
    train_val = benchmark['train_val']
    y_train_val = train_val.iloc[:, 2].tolist()

    fp_train_val = np.load(open("./features/" + name + "_train_val.npy", "rb"))
    fp_test = np.load(open("./features/" + name + "_test.npy", "rb"))
    
    xgb.fit(fp_train_val, y_train_val)
    pred_xgb = xgb.predict_proba(fp_test) if classification else xgb.predict(fp_test)
    
    return pred_xgb 
