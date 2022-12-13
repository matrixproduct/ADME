import numpy as np
import torch

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, \
                            roc_auc_score, precision_score, average_precision_score 
from scipy.stats import spearmanr

gpu_available = torch.cuda.is_available()


def get_problem_type(split):
    classification = False
    num_classes = None
    y = split['test']['Y'] 
    if isinstance(y[0], (np.int32, np.int64)):
        classification = True
        num_classes = len(y.value_counts())
    return classification, num_classes 

def dict_to_str(performance):
    performance_str = ',  '.join([f'{key}: {performance[key]:.4f}' for key in performance])
    return  performance_str


###############################################################################
# calculate various metrics of regression prediction





def spearman_metric(y_true, y_pred):
    """spearman metric
    """
    return spearmanr(y_true, y_pred)[0]


def regression_metrics(target, reg_prediction):
    
    
    assert len(target) == len(reg_prediction),\
        'target and reg_prediction  must have the same length'  
        
    target = np.array(target).reshape(-1, 1)    
    reg_prediction = np.array(reg_prediction).reshape(-1, 1)    
    
    
    reg_metrics = {}
    
    # RMSE
    reg_metrics['RMSE'] = mean_squared_error(target, reg_prediction, squared=False)
    
    # R^2
    reg_metrics['R^2'] = r2_score(target, reg_prediction)
    
    # MAE
    reg_metrics['MAE'] = mean_absolute_error(target, reg_prediction)
    
    
    return reg_metrics
    
###############################################################################
# calculate various metrics of classification prediction

def classification_metrics(target, prediction_prob, suppress_warnings=False, class_labels=None):
    
    assert len(target) == len(prediction_prob),\
        'target and prediction_prob must have the same length'  
        
    prediction_prob_ = np.array(prediction_prob)    
    m = prediction_prob_.shape[-1]  # number of classes    
    if not class_labels:
        class_labels = np.arange(m)  
    class_labels = np.array(class_labels)    
    
    clf_metrics = {}
    
    target_ = target.astype(int)   
    pred_ = class_labels[prediction_prob_.argmax(axis=1)]
    
    
    clf_metrics['accuracy'] = accuracy_score(target_, pred_) 
    
    
    #AUROC  
    try:
        if m == 2:
            clf_metrics['AUROC'] = roc_auc_score(target_, pred_)  
        else:
            clf_metrics['AUROC'] = roc_auc_score(target_,  prediction_prob_, multi_class='ovr') 
            
            
    except ValueError as e:
        if not suppress_warnings:
            print(f'\nWARNING: {e}\n')   
            # print(target_)
            # print(pred_)
    
    #AUPRC
    clf_metrics['AUPRC'] = average_precision_score(target_, pred_) 
            
        
    # Precision
    try:
        if m == 2:
            clf_metrics['Precision'] = precision_score(target_, pred_) 
        else:
            clf_metrics['Precision'] = precision_score(target_, pred_, average='micro') 
                       
    except ValueError as e:
        if not suppress_warnings:
            print(f'\nWARNING: {e}\n')   
            # print(target_)
            # print(pred_)
        
    # Positive fraction
    clf_metrics['Positive_fraction'] = len(target[target == 1]) / len(target)
   
    
    return clf_metrics