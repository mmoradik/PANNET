%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd
import random
from Utils import *
import matplotlib.pyplot as plt
%matplotlib inline
initialization()

Sunspot=pd.read_excel("Sunspot_series.xls", header=None)[1].values[0:-1]
Sc=MinMaxScaler(feature_range=(-.8, .8))
Sunspot_normed=Sc.fit_transform(Sunspot.reshape(-1, 1))
Data=Sunspot_normed.copy() # normalized data

n_training=221

y_a=Sc.inverse_transform(Data) #actual data

Train_orig=y_a[:n_training]  #y_a
Test_orig=y_a[n_training:]#Sunspot_normed


Error, MinFit,MaxFit,AvgFit,UNI,Train_predict,Test_predict,Network=Evolution(Data,y_a,n_training,Sc)
