# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:35:25 2020

@author:Woody
提交记录：
1在code的基础上对particle的坐标归一化（各个方向的坐标除以欧氏距离）
"""
import pandas as pd
from lightgbm import LGBMClassifier as lgb
df3 = pd.read_csv("train_jet.csv")
df4 = pd.read_csv("train_particle.csv")
df3_ = pd.read_csv("test_jet.csv")
df4_ = pd.read_csv("test_particle.csv")

df4['circle_particle_px'] = df4.apply(lambda x : x['particle_px']/x['particle_euclidean_distance'],axis = 1)
df4['circle_particle_py'] = df4.apply(lambda x : x['particle_py']/x['particle_euclidean_distance'],axis = 1)
df4['circle_particle_pz'] = df4.apply(lambda x : x['particle_pz']/x['particle_euclidean_distance'],axis = 1)

df4_['circle_particle_px'] = df4_.apply(lambda x : x['particle_px']/x['particle_euclidean_distance'],axis = 1)
df4_['circle_particle_py'] = df4_.apply(lambda x : x['particle_py']/x['particle_euclidean_distance'],axis = 1)
df4_['circle_particle_pz'] = df4_.apply(lambda x : x['particle_pz']/x['particle_euclidean_distance'],axis = 1)
############训练集
######circle_particle_px
mean_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].mean()
df3['mean_circle_particle_px'] = df3['jet_id'].apply(lambda x:mean_circle_particle_px[x])

max_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].max()
df3['max_circle_particle_px'] = df3['jet_id'].apply(lambda x:max_circle_particle_px[x])

min_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].min()
df3['min_circle_particle_px'] = df3['jet_id'].apply(lambda x:min_circle_particle_px[x])

median_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].median()
df3['median_circle_particle_px'] = df3['jet_id'].apply(lambda x:median_circle_particle_px[x])

sum_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].sum()
df3['sum_circle_particle_px'] = df3['jet_id'].apply(lambda x:sum_circle_particle_px[x])

std_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].std()
df3['std_circle_particle_px'] = df3['jet_id'].apply(lambda x:std_circle_particle_px[x])

var_circle_particle_px = df4.groupby(['jet_id'])['circle_particle_px'].var()
df3['var_circle_particle_px'] = df3['jet_id'].apply(lambda x:var_circle_particle_px[x])

######circle_particle_py
mean_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].mean()
df3['mean_circle_particle_py'] = df3['jet_id'].apply(lambda x:mean_circle_particle_py[x])

max_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].max()
df3['max_circle_particle_py'] = df3['jet_id'].apply(lambda x:max_circle_particle_py[x])

min_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].min()
df3['min_circle_particle_py'] = df3['jet_id'].apply(lambda x:min_circle_particle_py[x])

median_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].median()
df3['median_circle_particle_py'] = df3['jet_id'].apply(lambda x:median_circle_particle_py[x])

sum_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].sum()
df3['sum_circle_particle_py'] = df3['jet_id'].apply(lambda x:sum_circle_particle_py[x])

std_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].std()
df3['std_circle_particle_py'] = df3['jet_id'].apply(lambda x:std_circle_particle_py[x])

var_circle_particle_py = df4.groupby(['jet_id'])['circle_particle_py'].var()
df3['var_circle_particle_py'] = df3['jet_id'].apply(lambda x:var_circle_particle_py[x])

#######circle_particle_pz
mean_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].mean()
df3['mean_circle_particle_pz'] = df3['jet_id'].apply(lambda x:mean_circle_particle_pz[x])

max_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].max()
df3['max_circle_particle_pz'] = df3['jet_id'].apply(lambda x:max_circle_particle_pz[x])

min_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].min()
df3['min_circle_particle_pz'] = df3['jet_id'].apply(lambda x:min_circle_particle_pz[x])

median_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].median()
df3['median_circle_particle_pz'] = df3['jet_id'].apply(lambda x:median_circle_particle_pz[x])

sum_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].sum()
df3['sum_circle_particle_pz'] = df3['jet_id'].apply(lambda x:sum_circle_particle_pz[x])

std_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].std()
df3['std_circle_particle_pz'] = df3['jet_id'].apply(lambda x:std_circle_particle_pz[x])

var_circle_particle_pz = df4.groupby(['jet_id'])['circle_particle_pz'].var()
df3['var_circle_particle_pz'] = df3['jet_id'].apply(lambda x:var_circle_particle_pz[x])

############测试集
######circle_particle_px
mean_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].mean()
df3_['mean_circle_particle_px'] = df3_['jet_id'].apply(lambda x:mean_circle_particle_px[x])

max_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].max()
df3_['max_circle_particle_px'] = df3_['jet_id'].apply(lambda x:max_circle_particle_px[x])

min_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].min()
df3_['min_circle_particle_px'] = df3_['jet_id'].apply(lambda x:min_circle_particle_px[x])

median_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].median()
df3_['median_circle_particle_px'] = df3_['jet_id'].apply(lambda x:median_circle_particle_px[x])

sum_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].sum()
df3_['sum_circle_particle_px'] = df3_['jet_id'].apply(lambda x:sum_circle_particle_px[x])

std_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].std()
df3_['std_circle_particle_px'] = df3_['jet_id'].apply(lambda x:std_circle_particle_px[x])

var_circle_particle_px = df4_.groupby(['jet_id'])['circle_particle_px'].var()
df3_['var_circle_particle_px'] = df3_['jet_id'].apply(lambda x:var_circle_particle_px[x])

######circle_particle_py
mean_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].mean()
df3_['mean_circle_particle_py'] = df3_['jet_id'].apply(lambda x:mean_circle_particle_py[x])

max_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].max()
df3_['max_circle_particle_py'] = df3_['jet_id'].apply(lambda x:max_circle_particle_py[x])

min_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].min()
df3_['min_circle_particle_py'] = df3_['jet_id'].apply(lambda x:min_circle_particle_py[x])

median_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].median()
df3_['median_circle_particle_py'] = df3_['jet_id'].apply(lambda x:median_circle_particle_py[x])

sum_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].sum()
df3_['sum_circle_particle_py'] = df3_['jet_id'].apply(lambda x:sum_circle_particle_py[x])

std_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].std()
df3_['std_circle_particle_py'] = df3_['jet_id'].apply(lambda x:std_circle_particle_py[x])

var_circle_particle_py = df4_.groupby(['jet_id'])['circle_particle_py'].var()
df3_['var_circle_particle_py'] = df3_['jet_id'].apply(lambda x:var_circle_particle_py[x])

#######circle_particle_pz
mean_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].mean()
df3_['mean_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:mean_circle_particle_pz[x])

max_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].max()
df3_['max_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:max_circle_particle_pz[x])

min_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].min()
df3_['min_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:min_circle_particle_pz[x])

median_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].median()
df3_['median_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:median_circle_particle_pz[x])

sum_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].sum()
df3_['sum_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:sum_circle_particle_pz[x])

std_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].std()
df3_['std_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:std_circle_particle_pz[x])

var_circle_particle_pz = df4_.groupby(['jet_id'])['circle_particle_pz'].var()
df3_['var_circle_particle_pz'] = df3_['jet_id'].apply(lambda x:var_circle_particle_pz[x])

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
df3.to_csv("train_jet2.csv",index = False)
df4.to_csv("train_particle2.csv",index = False)
df3_.to_csv("test_jet2.csv",index = False)
df4_.to_csv("test_particle2.csv",index = False)