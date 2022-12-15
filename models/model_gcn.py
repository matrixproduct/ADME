"""
Graph Convolutional Network model build using PyTorch Geometric.
https://pytorch-geometric.readthedocs.io/en/latest/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from typing import Callable, List, Optional, Tuple


from tdc.single_pred import ADME
from tdc.utils import retrieve_dataset_names




import torch
import torch_geometric
from torch_geometric.datasets import Planetoid

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add
from typing_extensions import Literal, TypedDict
import torch_sparse
from torchmetrics import Accuracy


import utils
from utils import dict_to_str



# Graph Convolutional Network
class GCN(nn.Module):
    '''Graph Convolutional Network'''

    def __init__(
                self,
                num_node_features,
                hidden_dim=32,
                dropout_rate=0.5,
                classification=False,
                num_classes=None,
                ):

        super().__init__()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1 = GCNConv(num_node_features, hidden_dim)                       
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)                    
        self.dropout3 = nn.Dropout(dropout_rate)
        self.conv3 = GCNConv(hidden_dim, hidden_dim) 
        self.dropout4 = nn.Dropout(dropout_rate)
        self.classification = classification
        
        if classification:
            assert num_classes is not None, 'numeber of classes must be int'  
            self.lin  = nn.Linear(hidden_dim, num_classes)   
        else:
            self.lin  = nn.Linear(hidden_dim, 1) 
                   

    def forward(self, x, edge_index, batch):
        '''Forward pass'''
        
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.conv2(x, edge_index)
        # x = self.dropout3(x)
        # x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout4(x)
        x = self.lin(x)
        
            
        return x





#%% Train and test


# Stage = Literal['train', 'valid', 'test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_step(model, data_loader, optimizer, loss_fn):
    '''Train step'''
    
    model.train()
    train_loader = data_loader['train']
    loss_metric = {}
    
    for data in train_loader:  # iterate in batches over the training dataset
        data = data.to(device) # move to gpu if available
        optimizer.zero_grad()  # clear gradients
        logits = model(data.x, data.edge_index, data.batch)  # perform a single forward pass
        loss = loss_fn(logits, data.y) # compute the loss
        loss_metric['train_loss'] = loss.item()
        
        loss.backward()  # derive gradients
        optimizer.step()  # Update parameters based on gradients.
         
    
    return loss_metric


@torch.no_grad()
def eval_step(model, data_loader, loss_fn, stage):
    '''Evaluation step'''
    
    model.eval()
    loader = data_loader[stage]
    loss_metric = {}
    
    for data in loader:  # iterate in batches over the training dataset
        data = data.to(device) # move to gpu if available
        logits = model(data.x, data.edge_index, data.batch)  # perform a single forward pass
        loss = loss_fn(logits, data.y) # compute the loss
        loss_metric[f'{stage}_loss'] = loss.item()
    return loss_metric



def train(model, data, optimizer,
    loss_fn=None,
    max_epochs=200,
    early_stopping=10,
    print_interval=20,
    verbose=True):
    ''' Train and test'''
    
    if not loss_fn:
        if model.classification:
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.L1Loss()
    
    for epoch in range(max_epochs):
        loss_metric = train_step(model, data, optimizer, loss_fn)
        valid_loss_metric = eval_step(model, data, loss_fn, 'valid')
        total_loss_metric = {**loss_metric, **valid_loss_metric}
        if epoch == 0:
            history = {metric_key: [] for metric_key in total_loss_metric}
        for metric_key, metric_value in total_loss_metric.items():
            history[metric_key].append(metric_value)
        valid_loss = total_loss_metric['valid_loss']
        if epoch > early_stopping and valid_loss > np.mean(history['valid_loss'][-(early_stopping + 1) : -1]):
            if verbose:
                print('\nEarly stopping...')

            break

        if verbose and epoch % print_interval== 0:
            print(f'\nEpoch: {epoch}\n----------')
            print(f'{dict_to_str(loss_metric)}')
            print(f'{dict_to_str(valid_loss_metric)}')
           

    test_loss_metric = eval_step(model, data, loss_fn, 'test')
    if verbose:
        print(f'\nEpoch: {epoch}\n----------')
        print(f'{dict_to_str(loss_metric)}')
        print(f'{dict_to_str(valid_loss_metric)}')
        print(f'{dict_to_str(test_loss_metric)}')
        


    return history


def gcn_predict(split, **kwargs):
    ''' GCN model preditions'''
    
    classification, num_classes =  utils.get_problem_type(split)    
    loader = {}
    
    for key, df in split.items():
        smiles = df['Drug']
        targets = df['Y']
        
        
        graphs = []
        for smile, y in zip(smiles, targets):
            graph_data = from_smiles(smile)
            graph_data.x = graph_data.x.type(torch.float)
            graph_data.y = y 
            graphs.append(graph_data)
            
                       
        loader[key] = DataLoader(graphs, batch_size=100, shuffle=True)
    
    num_node_features = graphs[0].num_node_features

    SEED = 42
    MAX_EPOCHS = 200
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    EARLY_STOPPING = 10


    torch.manual_seed(SEED)
    model = GCN(num_node_features, classification=classification, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = train(model, loader, optimizer, max_epochs=MAX_EPOCHS, early_stopping=EARLY_STOPPING)

    outputs = {}
    with torch.no_grad():
      for stage in ['valid', 'test']:  
          output = []
          for data in loader[stage]:
              data = data.to(device) # move to gpu if available
              logits = model(data.x, data.edge_index, data.batch)  
              if model.classification:
                  prediction_prob = torch.nn.functional.softmax(logits, dim=1)
                  output.append(prediction_prob.detach().cpu().numpy()) 
              else:
                  output.append(logits.detach().cpu().numpy()) 
          outputs[stage] =  np.vstack(output)   
       
    return outputs['valid'], outputs['test']




     