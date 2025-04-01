### Environment Settings    
Here are the versions of some important Python libraries we used:
- pytorch 1.11.0
- numpy 1.20.3
- torch-geometric 1.7.2
- dgl-cu113 0.8.2
- scipy 1.7.1
- seaborn 0.11.2

### Datasets
We provide the datasets in the folder '.data/', and the data split file in the folder 'data/data_splits/'.

### Run the experiments
You can run the following commands directly.
#### Pre-training stage:
```
python pre_train.py --dataname cora 
```
The pre-trained pkl file should be saved in './pkl/'.

#### Downstram tuning stage (transductive learning):
```
python downstream_tune.py --pretrain_dataname cora --dataname cora --lr 0.005 --mode 0  --gpu 0
```

#### Downstram tuning stage (inductive learning), for example (wisconsin->texas):
```
python downstream_tune.py --pretrain_dataname wisconsin --dataname texas --lr 0.005 --mode 0  --gpu 0
```