# CSLSL


# Performances
The latest experimental results are as follows:

<table>
    <tr>
        <th> </th>
        <th colspan=3> Category </th>
        <th colspan=3> Location </th>
    </tr>
    <tr>
        <td></td>
        <td> R@1 </td>
        <td> R@5 </td>
        <td> R@10 </td>
        <td> R@1 </td>
        <td> R@5 </td>
        <td> R@10 </td>
    </tr>
    <tr>
        <td> NYC </td>
        <td> 0.327 </td>
        <td> 0.661 </td>
        <td> 0.759 </td>
        <td> 0.268 </td>
        <td> 0.568 </td>
        <td> 0.656 </td>
    </tr>
    <tr>
        <td> TKY </td>
        <td> 0.448 </td>
        <td> 0.801 </td>
        <td> 0.875 </td>
        <td> 0.240 </td>
        <td> 0.488 </td>
        <td> 0.580 </td>
    </tr>
    <tr>
        <td> Dallas </td>
        <td> - </td>
        <td> - </td>
        <td> - </td>
        <td> 0.126 </td>
        <td> 0.243 </td>
        <td> 0.297 </td>
    </tr>
</table>



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
1. Train and test a new model
> ```python
> python train_test.py --data_name NYC 
> ```

2. Evaluate a pretrained model
> ```python
> python evaluate.py --data_name NYC --model_name model_NYC
> ```

Detailed parameter description refers to ```evaluate.py``` and ```train_test.py```
