import pandas as pd
import numpy as np
import sklearn_pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import seaborn as sns
from pycox.evaluation import EvalSurv
sns.set_style("white")

import torch # For building the networks 
import torchtuples as tt # Some useful functions
from torch import nn
import torch.nn.functional as F

from pycox.datasets import support

import cox
from cox import CoxPH
#from pycox.evaluation import EvalSurv
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong

import model
from model import Cox_cluster

np.random.seed(42)
_ = torch.manual_seed(8)
data_path="path"
data1_train= pd.read_csv("path/data1_train.csv")
re_train=pd.read_csv("path/re_train.csv")
data1_test= pd.read_csv("path/data1_test.csv")
re_test=pd.read_csv("path/re_test.csv")
data1_val= pd.read_csv("path/data1_val.csv")
re_val=pd.read_csv("path/re_val.csv")
#data1.info()
#re.info()
df_test = data1_test
df_train = data1_train
df_val = data1_val
cols_std = []# numeric variables
cols_bin = ['x1', 'x2', 'x3','x4','x5','x6' ]  # binary variables
cols_cat = [] # categorical variables

#standardize = [([col], StandardScaler()) for col in cols_std]
leave = [(col, None) for col in cols_bin]
#categorical = [(col, OrderedCategoricalLong()) for col in cols_cat]
print(leave)
x_mapper = DataFrameMapper(leave)
#x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to convert data to 'int64'
#x_fit_transform = lambda df: tt.tuplefy(df)
#x_transform = lambda df: tt.tuplefy(df)
#x_fit_transform = lambda df: tt.tuplefy(x_mapper.fit_transform(df))
#x_transform = lambda df: tt.tuplefy(x_mapper.transform(df))
x_train = x_mapper.fit_transform(df_train).astype('float32')
#x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
#x_train = float('.'.join(str(elem) for elem in x_train[0]))
re_test = re_test
re_train = re_train
re_val=re_val
#x_mapper_long = DataFrameMapper(categorical)  # we need a separate mapper to convert data to 'int64'
#x_fit_transform = lambda df: tt.tuplefy(df)
#x_transform = lambda df: tt.tuplefy(df)
#x_fit_transform = lambda df: tt.tuplefy(x_mapper.fit_transform(df))
#x_transform = lambda df: tt.tuplefy(x_mapper.transform(df))
cols_bin_re=list(re_train.columns)[1:]
leave_re = [(col, None) for col in cols_bin_re]
#categorical = [(col, OrderedCategoricalLong()) for col in cols_cat]
print(leave_re)


x_mapper_re= DataFrameMapper(leave_re)
re_train = x_mapper_re.fit_transform(re_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
re_test = x_mapper_re.transform(re_test).astype('float32')
re_val = x_mapper_re.transform(re_val).astype('float32')
#print(re_train[0].shape)
get_target = lambda df: (df['o'].values, df['delta'].values)
y_train = get_target(df_train)
y_val = get_target(df_val)
durations_test, events_test = get_target(df_test)
val = x_val,re_val, y_val
print(durations_test)
#print(x_train[0].dtype)
#print(y_train[0].dtype)
#x_train=x_train.drop(columns=["split_train"])
optimizer = tt.optim.AdamWR(decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8,
                            cycle_multiplier=2)
In_Nodes= x_train.shape[1]
num_nodes = [64, 64]
Pathway_Nodes=num_nodes[0]
Hidden_Nodes=num_nodes[1]
Out_Nodes= 1
batch_norm = True
dropout = 0.2
output_bias = False
batch_size = 256

#net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
#                              dropout, output_bias=output_bias)
net= model.Cox_cluster(In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes)
model = cox.CoxPH(net, optimizer)
#print(model)


#model_2a.optimizer.set_lr(0.06)
epochs = 100
callbacks = [tt.callbacks.EarlyStopping()]
verbose = True

#log = model.fit(x_train, re_train,y_train, batch_size, epochs, callbacks, verbose)
batch_size = 256
lrfind = model.lr_finder(x_train, re_train,y_train,batch_size, tolerance=50)
#_ = lrfind.plot()
print(lrfind.get_best_lr())
#0.035111917342151515
model.optimizer.set_lr(0.011)
log = model.fit(x_train, re_train,y_train, batch_size, epochs, callbacks, verbose, val_data=val,
                val_batch_size=batch_size)
#_ = log.plot()
#print(model.partial_log_likelihood(*val).mean())
#print(np.squeeze(model.partial_log_likelihood(*val)[0],axis=0).shape)
#print(np.squeeze(model.partial_log_likelihood(*val)[0],axis=0)+model.partial_log_likelihood(*val)[1])
baseline = model.compute_baseline_hazards(x1=x_test, x2=re_test)
print(baseline.shape)
surv = model.predict_surv_df(x_test,re_test,baseline_hazards_=baseline)
#surv = model.predict_surv_df(x_test,re_test)
print(surv)
print(surv.index.values)
#ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
#C_INDEX=ev.concordance_td('antolini')
#print(C_INDEX)

#time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
#ev.brier_score(time_grid)
surv.flags
#np.isfortran(surv)
#surv = np.array(surv, order='C')
print(durations_test.shape[0])
print( surv.shape[1] )
print(surv.index.values.shape)

print(events_test.shape[0])

import concordance
c_index=concordance.concordance_td(durations_test, events_test,surv,surv.index.values)
print(c_index)

