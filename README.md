# Predictions for [ADME dataset](https://tdcommons.ai/single_pred_tasks/adme/) from the Therapeutics Data Commons (TDC).

## Installation

```bash
git clone https://github.com/matrixproduct/ADME
cd ADME
conda env create -f environment_adme.yml
conda activate adme
```
Download and extract features from [here](https://drive.google.com/file/d/1un1kO5ZoFQ6G7WCbL0SffTiYiBon06bT/view?usp=sharing). All the features files should be located in ./features/.

## Usage

Run `python benchmark_models.py` for models training and prediction. Change `set_names` in  `benchmark_models.py` to specify datasets for which the benchmarks should be calculated.

## Results

Results for each dataset can be found in ./results/.

## Models description

Three different types of models are employed for this project: Graph Convolutional Networks (GCNs), transformers and tree-like models.

1. GCN model is constructed from scratch using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html).
2. For transformers we use large pretrained models provided through [Simple Transformers](https://simpletransformers.ai/)
3. Tree models are represented by the XGBoost model trained on large number of features obtained from SMILES using many different methods. Here we follow the approach developed in 
['Accurate ADMET Prediction with XGBoost'](https://arxiv.org/abs/2204.07532v2) by Hao Tian, Rajas Ketkar, Peng Tao.

## Comments

The aim for this project was to demonstrate that I can build a workable ML pipeline rather than achieving high performance results. 
I chose three different approaches on purpose: building a model from scratch, training large pre-trained models and finally integrating a solution, which was designed for these specific datasets.     

I worked on the project during 5 days for several hours per day. I was familiar with all three types of models, but with different degrees of familiarity: tree like models have been used 
a lot in my current job, I have learnt recently about Graph Neural Networks and use GCN model in one project, I know about transformers for a long time, but I have never used them in my projects before.
The most time I spent investigating and choosing appropriate models in the case of transformers and tree like models and integrating them in my pipeline.

If I had more time I could make the following improvements:
1. GCN model: optimising architecture; hyperparameter tuning.
2. Transformers: investigating the literature to understand which pre-trained model would potentially work better; hyperparameter tuning.
3. XGBoost: try other tree-like models, such as LightGBM; hyperparameter tuning for classification tasks; include featurization for arbitrary datasets.
4. Pipeline: more flexibility in inputs and outputs; better representation of final results; possibility to set model parameters from the main script.
5. Pre-processing data: oversampling imbalanced datasets. 

## Conclusions

As the TDC web site provides the leaderboards for each dataset, it is reasonable to use the same metrics for comparing the models as the ones employed by TDC. 
I included those metrics, along with some additional metrics, in my comparison tables. One can see clearly that the XGBoost model outperforms the other two models. This is 
not surprising taking into account that the XGBoost model was optimised for the ADME datasets, while the other two models were not. Moreover the XGBoost model is ranked first in 
11 tasks and top 3 in 19 tasks in the TDC ADMET benchmark group.




