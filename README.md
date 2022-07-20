# CSLSL


# Datasets
- The processed data can be found in the "data" folder, which was processed by ```data_preproess.py``` and ```data_prepare.py```.
- The raw data can be found at the following open source.
    - [Foursqure (NYC and TKY)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset?authuser=0) 
    - [Gowalla (Dallas)](https://snap.stanford.edu/data/loc-gowalla.html)
    
# Requirements
- Python>=3.8
- Pytorch>=1.8.1
- Numpy
- Pandas


# Project Structure
- ```/data```: file to store processed data
- ```/results```: file to store results such as trained model and metrics.
- ```data_preprocess.py```: data preprocessing to filter sparse users and locations (fewer than 10 records) and merge consecutive records (same user and location on the same day).
- ```data_prepare.py```: data preparation for CSLSL (split trajectory and generate data).
- ```train_test.py```: the entry to train and test a new model.
- ```evaluate.py```: the entry to evalute a pretrained model.
- ```model.py```: model defination.
- ```utils.py```: tools such as batch generation and metric calculation.




# Usage
1. Evaluate a pretrained model
> ```python
> python evaluate.py --data_name NYC --model_name model_NYC
> ```

2. Train and test a new model
> ```python
> python train_test.py --data_name NYC 
> ```

Detailed parameter description refers to ```evaluate.py``` and ```train_test.py```
