# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:01:37 2020

@author: Woody
提交记录:
1在code2的基础上，加了相同event下jet的坐标归一化特征
  之前的几次提交，成绩从0.6821到0.6886，有小小的提升
"""
import pandas as pd
from lightgbm import LGBMClassifier as lgb
df3 = pd.read_csv("train_jet2.csv")
df4 = pd.read_csv("train_particle2.csv")
df3_ = pd.read_csv("test_jet2.csv")
df4_ = pd.read_csv("test_particle2.csv")


df3['circle_jet_px'] = df3.apply(lambda x : x['jet_px']/x['jet_euclidean_distance'],axis = 1)
df3['circle_jet_py'] = df3.apply(lambda x : x['jet_py']/x['jet_euclidean_distance'],axis = 1)
df3['circle_jet_pz'] = df3.apply(lambda x : x['jet_pz']/x['jet_euclidean_distance'],axis = 1)

df3_['circle_jet_px'] = df3_.apply(lambda x : x['jet_px']/x['jet_euclidean_distance'],axis = 1)
df3_['circle_jet_py'] = df3_.apply(lambda x : x['jet_py']/x['jet_euclidean_distance'],axis = 1)
df3_['circle_jet_pz'] = df3_.apply(lambda x : x['jet_pz']/x['jet_euclidean_distance'],axis = 1)

####训练集
######circle_jet_px
mean_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].mean()
df3['mean_circle_jet_px'] = df3['event_id'].apply(lambda x:mean_circle_jet_px[x])

max_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].max()
df3['max_circle_jet_px'] = df3['event_id'].apply(lambda x:max_circle_jet_px[x])

min_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].min()
df3['min_circle_jet_px'] = df3['event_id'].apply(lambda x:min_circle_jet_px[x])

median_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].median()
df3['median_circle_jet_px'] = df3['event_id'].apply(lambda x:median_circle_jet_px[x])

sum_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].sum()
df3['sum_circle_jet_px'] = df3['event_id'].apply(lambda x:sum_circle_jet_px[x])

std_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].std()
df3['std_circle_jet_px'] = df3['event_id'].apply(lambda x:std_circle_jet_px[x])

var_circle_jet_px = df3.groupby(['event_id'])['circle_jet_px'].var()
df3['var_circle_jet_px'] = df3['event_id'].apply(lambda x:var_circle_jet_px[x])
######circle_jet_py
mean_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].mean()
df3['mean_circle_jet_py'] = df3['event_id'].apply(lambda x:mean_circle_jet_py[x])

max_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].max()
df3['max_circle_jet_py'] = df3['event_id'].apply(lambda x:max_circle_jet_py[x])

min_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].min()
df3['min_circle_jet_py'] = df3['event_id'].apply(lambda x:min_circle_jet_py[x])

median_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].median()
df3['median_circle_jet_py'] = df3['event_id'].apply(lambda x:median_circle_jet_py[x])

sum_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].sum()
df3['sum_circle_jet_py'] = df3['event_id'].apply(lambda x:sum_circle_jet_py[x])

std_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].std()
df3['std_circle_jet_py'] = df3['event_id'].apply(lambda x:std_circle_jet_py[x])

var_circle_jet_py = df3.groupby(['event_id'])['circle_jet_py'].var()
df3['var_circle_jet_py'] = df3['event_id'].apply(lambda x:var_circle_jet_py[x])
######circle_jet_pz
mean_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].mean()
df3['mean_circle_jet_pz'] = df3['event_id'].apply(lambda x:mean_circle_jet_pz[x])

max_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].max()
df3['max_circle_jet_pz'] = df3['event_id'].apply(lambda x:max_circle_jet_pz[x])

min_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].min()
df3['min_circle_jet_pz'] = df3['event_id'].apply(lambda x:min_circle_jet_pz[x])

median_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].median()
df3['median_circle_jet_pz'] = df3['event_id'].apply(lambda x:median_circle_jet_pz[x])

sum_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].sum()
df3['sum_circle_jet_pz'] = df3['event_id'].apply(lambda x:sum_circle_jet_pz[x])

std_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].std()
df3['std_circle_jet_pz'] = df3['event_id'].apply(lambda x:std_circle_jet_pz[x])

var_circle_jet_pz = df3.groupby(['event_id'])['circle_jet_pz'].var()
df3['var_circle_jet_z'] = df3['event_id'].apply(lambda x:var_circle_jet_pz[x])

####测试集
######circle_jet_px
mean_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].mean()
df3_['mean_circle_jet_px'] = df3_['event_id'].apply(lambda x:mean_circle_jet_px[x])

max_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].max()
df3_['max_circle_jet_px'] = df3_['event_id'].apply(lambda x:max_circle_jet_px[x])

min_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].min()
df3_['min_circle_jet_px'] = df3_['event_id'].apply(lambda x:min_circle_jet_px[x])

median_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].median()
df3_['median_circle_jet_px'] = df3_['event_id'].apply(lambda x:median_circle_jet_px[x])

sum_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].sum()
df3_['sum_circle_jet_px'] = df3_['event_id'].apply(lambda x:sum_circle_jet_px[x])

std_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].std()
df3_['std_circle_jet_px'] = df3_['event_id'].apply(lambda x:std_circle_jet_px[x])

var_circle_jet_px = df3_.groupby(['event_id'])['circle_jet_px'].var()
df3_['var_circle_jet_px'] = df3_['event_id'].apply(lambda x:var_circle_jet_px[x])
######circle_jet_py
mean_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].mean()
df3_['mean_circle_jet_py'] = df3_['event_id'].apply(lambda x:mean_circle_jet_py[x])

max_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].max()
df3_['max_circle_jet_py'] = df3_['event_id'].apply(lambda x:max_circle_jet_py[x])

min_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].min()
df3_['min_circle_jet_py'] = df3_['event_id'].apply(lambda x:min_circle_jet_py[x])

median_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].median()
df3_['median_circle_jet_py'] = df3_['event_id'].apply(lambda x:median_circle_jet_py[x])

sum_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].sum()
df3_['sum_circle_jet_py'] = df3_['event_id'].apply(lambda x:sum_circle_jet_py[x])

std_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].std()
df3_['std_circle_jet_py'] = df3_['event_id'].apply(lambda x:std_circle_jet_py[x])

var_circle_jet_py = df3_.groupby(['event_id'])['circle_jet_py'].var()
df3_['var_circle_jet_py'] = df3_['event_id'].apply(lambda x:var_circle_jet_py[x])
######circle_jet_pz
mean_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].mean()
df3_['mean_circle_jet_pz'] = df3_['event_id'].apply(lambda x:mean_circle_jet_pz[x])

max_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].max()
df3_['max_circle_jet_pz'] = df3_['event_id'].apply(lambda x:max_circle_jet_pz[x])

min_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].min()
df3_['min_circle_jet_pz'] = df3_['event_id'].apply(lambda x:min_circle_jet_pz[x])

median_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].median()
df3_['median_circle_jet_pz'] = df3_['event_id'].apply(lambda x:median_circle_jet_pz[x])

sum_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].sum()
df3_['sum_circle_jet_pz'] = df3_['event_id'].apply(lambda x:sum_circle_jet_pz[x])

std_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].std()
df3_['std_circle_jet_pz'] = df3_['event_id'].apply(lambda x:std_circle_jet_pz[x])

var_circle_jet_pz = df3_.groupby(['event_id'])['circle_jet_pz'].var()
df3_['var_circle_jet_z'] = df3_['event_id'].apply(lambda x:var_circle_jet_pz[x])

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
df3.to_csv("train_jet3.csv",index = False)
df4.to_csv("train_particle3.csv",index = False)
df3_.to_csv("test_jet3.csv",index = False)
df4_.to_csv("test_particle3.csv",index = False)