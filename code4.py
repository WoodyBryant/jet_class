# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:43:34 2020

@author: Woody
提交记录
1 在code3的基础上加了particle在各个坐标轴上的动量分量
  成绩从0.6886到0.6888
"""
import pandas as pd
from lightgbm import LGBMClassifier as lgb
df3 = pd.read_csv("train_jet3.csv")
df4 = pd.read_csv("train_particle3.csv")
df3_ = pd.read_csv("test_jet3.csv")
df4_ = pd.read_csv("test_particle3.csv")

df4['momentum_particle_px'] = df4.apply(lambda x : x['circle_particle_px']*x['particle_momentum'],axis = 1)
df4['momentum_particle_py'] = df4.apply(lambda x : x['circle_particle_py']*x['particle_momentum'],axis = 1)
df4['momentum_particle_pz'] = df4.apply(lambda x : x['circle_particle_pz']*x['particle_momentum'],axis = 1)

df4_['momentum_particle_px'] = df4_.apply(lambda x : x['circle_particle_px']*x['particle_momentum'],axis = 1)
df4_['momentum_particle_py'] = df4_.apply(lambda x : x['circle_particle_py']*x['particle_momentum'],axis = 1)
df4_['momentum_particle_pz'] = df4_.apply(lambda x : x['circle_particle_pz']*x['particle_momentum'],axis = 1)

############训练集
######momentum_particle_px
mean_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].mean()
df3['mean_momentum_particle_px'] = df3['jet_id'].apply(lambda x:mean_momentum_particle_px[x])

max_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].max()
df3['max_momentum_particle_px'] = df3['jet_id'].apply(lambda x:max_momentum_particle_px[x])

min_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].min()
df3['min_momentum_particle_px'] = df3['jet_id'].apply(lambda x:min_momentum_particle_px[x])

median_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].median()
df3['median_momentum_particle_px'] = df3['jet_id'].apply(lambda x:median_momentum_particle_px[x])

sum_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].sum()
df3['sum_momentum_particle_px'] = df3['jet_id'].apply(lambda x:sum_momentum_particle_px[x])

std_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].std()
df3['std_momentum_particle_px'] = df3['jet_id'].apply(lambda x:std_momentum_particle_px[x])

var_momentum_particle_px = df4.groupby(['jet_id'])['momentum_particle_px'].var()
df3['var_momentum_particle_px'] = df3['jet_id'].apply(lambda x:var_momentum_particle_px[x])

######momentum_particle_py
mean_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].mean()
df3['mean_momentum_particle_py'] = df3['jet_id'].apply(lambda x:mean_momentum_particle_py[x])

max_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].max()
df3['max_momentum_particle_py'] = df3['jet_id'].apply(lambda x:max_momentum_particle_py[x])

min_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].min()
df3['min_momentum_particle_py'] = df3['jet_id'].apply(lambda x:min_momentum_particle_py[x])

median_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].median()
df3['median_momentum_particle_py'] = df3['jet_id'].apply(lambda x:median_momentum_particle_py[x])

sum_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].sum()
df3['sum_momentum_particle_py'] = df3['jet_id'].apply(lambda x:sum_momentum_particle_py[x])

std_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].std()
df3['std_momentum_particle_py'] = df3['jet_id'].apply(lambda x:std_momentum_particle_py[x])

var_momentum_particle_py = df4.groupby(['jet_id'])['momentum_particle_py'].var()
df3['var_momentum_particle_py'] = df3['jet_id'].apply(lambda x:var_momentum_particle_py[x])


#######cmomentum_particle_pz'
mean_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].mean()
df3['mean_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:mean_momentum_particle_pz[x])

max_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].max()
df3['max_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:max_momentum_particle_pz[x])

min_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].min()
df3['min_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:min_momentum_particle_pz[x])

median_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].median()
df3['median_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:median_momentum_particle_pz[x])

sum_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].sum()
df3['sum_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:sum_momentum_particle_pz[x])

std_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].std()
df3['std_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:std_momentum_particle_pz[x])

var_momentum_particle_pz = df4.groupby(['jet_id'])['momentum_particle_pz'].var()
df3['var_momentum_particle_pz'] = df3['jet_id'].apply(lambda x:var_momentum_particle_pz[x])

############测试集
######momentum_particle_px
mean_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].mean()
df3_['mean_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:mean_momentum_particle_px[x])

max_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].max()
df3_['max_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:max_momentum_particle_px[x])

min_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].min()
df3_['min_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:min_momentum_particle_px[x])

median_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].median()
df3_['median_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:median_momentum_particle_px[x])

sum_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].sum()
df3_['sum_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:sum_momentum_particle_px[x])

std_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].std()
df3_['std_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:std_momentum_particle_px[x])

var_momentum_particle_px = df4_.groupby(['jet_id'])['momentum_particle_px'].var()
df3_['var_momentum_particle_px'] = df3_['jet_id'].apply(lambda x:var_momentum_particle_px[x])

######cmomentum_particle_py
mean_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].mean()
df3_['mean_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:mean_momentum_particle_py[x])

max_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].max()
df3_['max_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:max_momentum_particle_py[x])

min_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].min()
df3_['min_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:min_momentum_particle_py[x])

median_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].median()
df3_['median_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:median_momentum_particle_py[x])

sum_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].sum()
df3_['sum_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:sum_momentum_particle_py[x])

std_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].std()
df3_['std_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:std_momentum_particle_py[x])

var_momentum_particle_py = df4_.groupby(['jet_id'])['momentum_particle_py'].var()
df3_['var_momentum_particle_py'] = df3_['jet_id'].apply(lambda x:var_momentum_particle_py[x])


#######momentum_particle_pz
mean_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].mean()
df3_['mean_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:mean_momentum_particle_pz[x])

max_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].max()
df3_['max_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:max_momentum_particle_pz[x])

min_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].min()
df3_['min_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:min_momentum_particle_pz[x])

median_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].median()
df3_['median_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:median_momentum_particle_pz[x])

sum_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].sum()
df3_['sum_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:sum_momentum_particle_pz[x])

std_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].std()
df3_['std_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:std_momentum_particle_pz[x])

var_momentum_particle_pz = df4_.groupby(['jet_id'])['momentum_particle_pz'].var()
df3_['var_momentum_particle_pz'] = df3_['jet_id'].apply(lambda x:var_momentum_particle_pz[x])

features = df3.columns
features = list(features)
features.remove('jet_id')
features.remove('event_id')
features.remove('label')

model = lgb()
y_predict = model.fit(df3[features],df3['label']).predict(df3_[features])
df5 = pd.DataFrame()
df5['id'] = df3_['jet_id']
df5['label'] = y_predict
df5.to_csv("submit.csv",index = False)
df3.to_csv("train_jet4.csv",index = False)
df4.to_csv("train_particle4.csv",index = False)
df3_.to_csv("test_jet4.csv",index = False)
df4_.to_csv("test_particle4.csv",index = False)
