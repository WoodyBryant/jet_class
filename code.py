# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:02:56 2020

@author: Woody
提交记录:
1 不加特征，只把jet对应的event下包含的jet数加上，得分为0.603
2 在1的基础上加一些particle的统计特征，比如均值，最大值，最小值，中位数，得分0.625
3 在1的基础上增加了欧氏距离
4 jet的平均速度是有害特征
5 同一jet中的particle的欧式距离的均值是有用特征
6 jet的动量是有害特征
7 极差没有作用，因为极差是最大值和最小值的线性组合
8 在2的基础上加入了求和，方差，标准差，得分0.62756
9 在8的基础上加入了同一event下的统计特征，得分0.68218
10 加入粒子的动量的统计特征，还没提交
准备尝试方向:
1 利用测试集来为添加特征
2 多添加物理特征
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
##导入数据

#导入训练集数据
df1 = pd.read_csv("complex_train_R04_jet.csv")
df2 = pd.read_csv("complex_train_R04_event.csv")
df3 = pd.merge(df1,df2,on = ['event_id'],how = 'left')
df4 = pd.read_csv("complex_train_R04_particle.csv")
#导入测试集数据
df1_ = pd.read_csv("complex_test_R04_jet.csv")
df2_ = pd.read_csv("complex_test_R04_event.csv")
df3_ = pd.merge(df1_,df2_,on = ['event_id'],how = 'left')
df4_ = pd.read_csv("complex_test_R04_particle.csv")


####################################对训练集集进行预处理############################
#####物理特征

#jet以及particle的欧氏距离
df3['jet_euclidean_distance'] = df3.apply(lambda x:(x['jet_px']**2+x['jet_py']**2+x['jet_pz']**2)**0.5,axis = 1)   
df4['particle_euclidean_distance'] = df4.apply(lambda x:(x['particle_px']**2+x['particle_py']**2+x['particle_pz']**2)**0.5,axis = 1)   

##particle的动量
df4['particle_momentum'] = df4.apply(lambda x:(2*x['particle_energy']*x['particle_mass'])**0.5,axis = 1)

#particle坐标xyz,以及energy、mass,欧氏距离的一些统计特征
#同一event下的jet的统计特征
     

##均值
mean_particle_px = df4.groupby(['jet_id'])['particle_px'].mean()
df3['mean_particle_px'] = df3['jet_id'].apply(lambda x:mean_particle_px[x])

mean_particle_py = df4.groupby(['jet_id'])['particle_py'].mean()
df3['mean_particle_py'] = df3['jet_id'].apply(lambda x:mean_particle_py[x])

mean_particle_pz = df4.groupby(['jet_id'])['particle_pz'].mean()
df3['mean_particle_pz'] = df3['jet_id'].apply(lambda x:mean_particle_pz[x])

mean_particle_energy = df4.groupby(['jet_id'])['particle_energy'].mean()
df3['mean_particle_energy'] = df3['jet_id'].apply(lambda x:mean_particle_energy[x])

mean_particle_mass = df4.groupby(['jet_id'])['particle_mass'].mean()
df3['mean_particle_mass'] = df3['jet_id'].apply(lambda x:mean_particle_mass[x])

mean_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].mean()
df3['mean_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:mean_particle_euclidean_distance[x])

mean_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].mean()
df3['mean_particle_momentum'] = df3['jet_id'].apply(lambda x:mean_particle_momentum[x])
##
mean_jet_px = df3.groupby(['event_id'])['jet_px'].mean()
df3['mean_jet_px'] = df3['event_id'].apply(lambda x:mean_jet_px[x])

mean_jet_py = df3.groupby(['event_id'])['jet_py'].mean()
df3['mean_jet_py'] = df3['event_id'].apply(lambda x:mean_jet_py[x])

mean_jet_pz = df3.groupby(['event_id'])['jet_pz'].mean()
df3['mean_jet_pz'] = df3['event_id'].apply(lambda x:mean_jet_pz[x])

mean_jet_energy = df3.groupby(['event_id'])['jet_energy'].mean()
df3['mean_jet_energy'] = df3['event_id'].apply(lambda x:mean_jet_energy[x])

mean_jet_mass = df3.groupby(['event_id'])['jet_mass'].mean()
df3['mean_jet_mass'] = df3['event_id'].apply(lambda x:mean_jet_mass[x])

mean_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].mean()
df3['mean_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:mean_jet_euclidean_distance[x])

##最大值
max_particle_px = df4.groupby(['jet_id'])['particle_px'].max()
df3['max_particle_px'] = df3['jet_id'].apply(lambda x:max_particle_px[x])

max_particle_py = df4.groupby(['jet_id'])['particle_py'].max()
df3['max_particle_py'] = df3['jet_id'].apply(lambda x:max_particle_py[x])

max_particle_pz = df4.groupby(['jet_id'])['particle_pz'].max()
df3['max_particle_pz'] = df3['jet_id'].apply(lambda x:max_particle_pz[x])

max_particle_energy = df4.groupby(['jet_id'])['particle_energy'].max()
df3['max_particle_energy'] = df3['jet_id'].apply(lambda x:max_particle_energy[x])

max_particle_mass = df4.groupby(['jet_id'])['particle_mass'].max()
df3['max_particle_mass'] = df3['jet_id'].apply(lambda x:max_particle_mass[x])

max_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].mean()
df3['max_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:max_particle_euclidean_distance[x])

max_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].max()
df3['max_particle_momentum'] = df3['jet_id'].apply(lambda x:max_particle_momentum[x])
##
max_jet_px = df3.groupby(['event_id'])['jet_px'].max()
df3['max_jet_px'] = df3['event_id'].apply(lambda x:max_jet_px[x])

max_jet_py = df3.groupby(['event_id'])['jet_py'].max()
df3['max_jet_py'] = df3['event_id'].apply(lambda x:max_jet_py[x])

max_jet_pz = df3.groupby(['event_id'])['jet_pz'].max()
df3['max_jet_pz'] = df3['event_id'].apply(lambda x:max_jet_pz[x])

max_jet_energy = df3.groupby(['event_id'])['jet_energy'].max()
df3['max_jet_energy'] = df3['event_id'].apply(lambda x:max_jet_energy[x])

max_jet_mass = df3.groupby(['event_id'])['jet_mass'].max()
df3['max_jet_mass'] = df3['event_id'].apply(lambda x:max_jet_mass[x])

max_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].max()
df3['max_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:max_jet_euclidean_distance[x])

##最小值
min_particle_px = df4.groupby(['jet_id'])['particle_px'].min()
df3['min_particle_px'] = df3['jet_id'].apply(lambda x:min_particle_px[x])

min_particle_py = df4.groupby(['jet_id'])['particle_py'].min()
df3['min_particle_py'] = df3['jet_id'].apply(lambda x:min_particle_py[x])

min_particle_pz = df4.groupby(['jet_id'])['particle_pz'].min()
df3['min_particle_pz'] = df3['jet_id'].apply(lambda x:min_particle_pz[x])

min_particle_energy = df4.groupby(['jet_id'])['particle_energy'].min()
df3['min_particle_energy'] = df3['jet_id'].apply(lambda x:min_particle_energy[x])

min_particle_mass = df4.groupby(['jet_id'])['particle_mass'].min()
df3['min_particle_mass'] = df3['jet_id'].apply(lambda x:min_particle_mass[x])

min_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].mean()
df3['min_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:min_particle_euclidean_distance[x])

min_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].min()
df3['min_particle_momentum'] = df3['jet_id'].apply(lambda x:min_particle_momentum[x])

##
min_jet_px = df3.groupby(['event_id'])['jet_px'].min()
df3['min_jet_px'] = df3['event_id'].apply(lambda x:min_jet_px[x])

min_jet_py = df3.groupby(['event_id'])['jet_py'].min()
df3['min_jet_py'] = df3['event_id'].apply(lambda x:min_jet_py[x])

min_jet_pz = df3.groupby(['event_id'])['jet_pz'].min()
df3['min_jet_pz'] = df3['event_id'].apply(lambda x:min_jet_pz[x])

min_jet_energy = df3.groupby(['event_id'])['jet_energy'].min()
df3['min_jet_energy'] = df3['event_id'].apply(lambda x:min_jet_energy[x])

min_jet_mass = df3.groupby(['event_id'])['jet_mass'].min()
df3['min_jet_mass'] = df3['event_id'].apply(lambda x:min_jet_mass[x])

min_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].min()
df3['min_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:min_jet_euclidean_distance[x])

##中位数
median_particle_px = df4.groupby(['jet_id'])['particle_px'].median()
df3['median_particle_px'] = df3['jet_id'].apply(lambda x:median_particle_px[x])

median_particle_py = df4.groupby(['jet_id'])['particle_py'].median()
df3['median_particle_py'] = df3['jet_id'].apply(lambda x:median_particle_py[x])

median_particle_pz = df4.groupby(['jet_id'])['particle_pz'].median()
df3['median_particle_pz'] = df3['jet_id'].apply(lambda x:median_particle_pz[x])

median_particle_energy = df4.groupby(['jet_id'])['particle_energy'].median()
df3['median_particle_energy'] = df3['jet_id'].apply(lambda x:median_particle_energy[x])

median_particle_mass = df4.groupby(['jet_id'])['particle_mass'].median()
df3['median_particle_mass'] = df3['jet_id'].apply(lambda x:median_particle_mass[x])

median_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].mean()
df3['median_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:median_particle_euclidean_distance[x])

median_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].median()
df3['median_particle_momentum'] = df3['jet_id'].apply(lambda x:median_particle_momentum[x])
##
median_jet_px = df3.groupby(['event_id'])['jet_px'].median()
df3['median_jet_px'] = df3['event_id'].apply(lambda x:median_jet_px[x])

median_jet_py = df3.groupby(['event_id'])['jet_py'].median()
df3['median_jet_py'] = df3['event_id'].apply(lambda x:median_jet_py[x])

median_jet_pz = df3.groupby(['event_id'])['jet_pz'].median()
df3['median_jet_pz'] = df3['event_id'].apply(lambda x:median_jet_pz[x])

median_jet_energy = df3.groupby(['event_id'])['jet_energy'].median()
df3['median_jet_energy'] = df3['event_id'].apply(lambda x:median_jet_energy[x])

median_jet_mass = df3.groupby(['event_id'])['jet_mass'].median()
df3['median_jet_mass'] = df3['event_id'].apply(lambda x:median_jet_mass[x])

median_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].median()
df3['median_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:median_jet_euclidean_distance[x])

##求和
sum_particle_px = df4.groupby(['jet_id'])['particle_px'].sum()
df3['sum_particle_px'] = df3['jet_id'].apply(lambda x:sum_particle_px[x])

sum_particle_py = df4.groupby(['jet_id'])['particle_py'].sum()
df3['sum_particle_py'] = df3['jet_id'].apply(lambda x:sum_particle_py[x])

sum_particle_pz = df4.groupby(['jet_id'])['particle_pz'].sum()
df3['sum_particle_pz'] = df3['jet_id'].apply(lambda x:sum_particle_pz[x])

sum_particle_energy = df4.groupby(['jet_id'])['particle_energy'].sum()
df3['sum_particle_energy'] = df3['jet_id'].apply(lambda x:sum_particle_energy[x])

sum_particle_mass = df4.groupby(['jet_id'])['particle_mass'].sum()
df3['sum_particle_mass'] = df3['jet_id'].apply(lambda x:sum_particle_mass[x])

sum_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].sum()
df3['sum_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:sum_particle_euclidean_distance[x])

sum_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].sum()
df3['sum_particle_momentum'] = df3['jet_id'].apply(lambda x:sum_particle_momentum[x])
##
sum_jet_px = df3.groupby(['event_id'])['jet_px'].sum()
df3['sum_jet_px'] = df3['event_id'].apply(lambda x:sum_jet_px[x])

sum_jet_py = df3.groupby(['event_id'])['jet_py'].sum()
df3['sum_jet_py'] = df3['event_id'].apply(lambda x:sum_jet_py[x])

sum_jet_pz = df3.groupby(['event_id'])['jet_pz'].sum()
df3['sum_jet_pz'] = df3['event_id'].apply(lambda x:sum_jet_pz[x])

sum_jet_energy = df3.groupby(['event_id'])['jet_energy'].sum()
df3['sum_jet_energy'] = df3['event_id'].apply(lambda x:sum_jet_energy[x])

sum_jet_mass = df3.groupby(['event_id'])['jet_mass'].sum()
df3['sum_jet_mass'] = df3['event_id'].apply(lambda x:sum_jet_mass[x])

sum_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].sum()
df3['sum_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:sum_jet_euclidean_distance[x])

##标准差
std_particle_px = df4.groupby(['jet_id'])['particle_px'].std()
df3['std_particle_px'] = df3['jet_id'].apply(lambda x:std_particle_px[x])

std_particle_py = df4.groupby(['jet_id'])['particle_py'].std()
df3['std_particle_py'] = df3['jet_id'].apply(lambda x:std_particle_py[x])

std_particle_pz = df4.groupby(['jet_id'])['particle_pz'].std()
df3['std_particle_pz'] = df3['jet_id'].apply(lambda x:std_particle_pz[x])

std_particle_energy = df4.groupby(['jet_id'])['particle_energy'].std()
df3['std_particle_energy'] = df3['jet_id'].apply(lambda x:std_particle_energy[x])

std_particle_mass = df4.groupby(['jet_id'])['particle_mass'].std()
df3['std_particle_mass'] = df3['jet_id'].apply(lambda x:std_particle_mass[x])

std_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].std()
df3['std_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:std_particle_euclidean_distance[x])

std_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].std()
df3['std_particle_momentum'] = df3['jet_id'].apply(lambda x:std_particle_momentum[x])
##
std_jet_px = df3.groupby(['event_id'])['jet_px'].std()
df3['std_jet_px'] = df3['event_id'].apply(lambda x:std_jet_px[x])

std_jet_py = df3.groupby(['event_id'])['jet_py'].std()
df3['std_jet_py'] = df3['event_id'].apply(lambda x:std_jet_py[x])

std_jet_pz = df3.groupby(['event_id'])['jet_pz'].std()
df3['std_jet_pz'] = df3['event_id'].apply(lambda x:std_jet_pz[x])

std_jet_energy = df3.groupby(['event_id'])['jet_energy'].std()
df3['std_jet_energy'] = df3['event_id'].apply(lambda x:std_jet_energy[x])

std_jet_mass = df3.groupby(['event_id'])['jet_mass'].std()
df3['std_jet_mass'] = df3['event_id'].apply(lambda x:std_jet_mass[x])

std_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].std()
df3['std_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:std_jet_euclidean_distance[x])

##方差
var_particle_px = df4.groupby(['jet_id'])['particle_px'].var()
df3['var_particle_px'] = df3['jet_id'].apply(lambda x:var_particle_px[x])

var_particle_py = df4.groupby(['jet_id'])['particle_py'].var()
df3['var_particle_py'] = df3['jet_id'].apply(lambda x:var_particle_py[x])

var_particle_pz = df4.groupby(['jet_id'])['particle_pz'].var()
df3['var_particle_pz'] = df3['jet_id'].apply(lambda x:var_particle_pz[x])

var_particle_energy = df4.groupby(['jet_id'])['particle_energy'].var()
df3['var_particle_energy'] = df3['jet_id'].apply(lambda x:var_particle_energy[x])

var_particle_mass = df4.groupby(['jet_id'])['particle_mass'].var()
df3['var_particle_mass'] = df3['jet_id'].apply(lambda x:var_particle_mass[x])

var_particle_euclidean_distance = df4.groupby(['jet_id'])['particle_euclidean_distance'].var()
df3['var_particle_euclidean_distance'] = df3['jet_id'].apply(lambda x:var_particle_euclidean_distance[x])

var_particle_momentum = df4.groupby(['jet_id'])['particle_momentum'].var()
df3['var_particle_momentum'] = df3['jet_id'].apply(lambda x:var_particle_momentum[x])
##
var_jet_px = df3.groupby(['event_id'])['jet_px'].var()
df3['var_jet_px'] = df3['event_id'].apply(lambda x:var_jet_px[x])

var_jet_py = df3.groupby(['event_id'])['jet_py'].var()
df3['var_jet_py'] = df3['event_id'].apply(lambda x:var_jet_py[x])

var_jet_pz = df3.groupby(['event_id'])['jet_pz'].var()
df3['var_jet_pz'] = df3['event_id'].apply(lambda x:var_jet_pz[x])

var_jet_energy = df3.groupby(['event_id'])['jet_energy'].var()
df3['var_jet_energy'] = df3['event_id'].apply(lambda x:var_jet_energy[x])

var_jet_mass = df3.groupby(['event_id'])['jet_mass'].var()
df3['var_jet_mass'] = df3['event_id'].apply(lambda x:var_jet_mass[x])

var_jet_euclidean_distance = df3.groupby(['event_id'])['jet_euclidean_distance'].var()
df3['var_jet_euclidean_distance'] = df3['event_id'].apply(lambda x:var_jet_euclidean_distance[x])

####################################对测试集进行预处理############################
#
###物理特征

#欧式距离
df3_['jet_euclidean_distance'] = df3_.apply(lambda x:(x['jet_px']**2+x['jet_py']**2+x['jet_pz']**2)**0.5,axis = 1) 
df4_['particle_euclidean_distance'] = df4_.apply(lambda x:(x['particle_px']**2+x['particle_py']**2+x['particle_pz']**2)**0.5,axis = 1)    

##particle的动量
df4_['particle_momentum'] = df4_.apply(lambda x:(2*x['particle_energy']*x['particle_mass'])**0.5,axis = 1)

##particle坐标xyz,以及energy、mass的一些统计特征
#
##均值
mean_particle_px = df4_.groupby(['jet_id'])['particle_px'].mean()
df3_['mean_particle_px'] = df3_['jet_id'].apply(lambda x:mean_particle_px[x])
#
mean_particle_py = df4_.groupby(['jet_id'])['particle_py'].mean()
df3_['mean_particle_py'] = df3_['jet_id'].apply(lambda x:mean_particle_py[x])

mean_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].mean()
df3_['mean_particle_pz'] = df3_['jet_id'].apply(lambda x:mean_particle_pz[x])

mean_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].mean()
df3_['mean_particle_energy'] = df3_['jet_id'].apply(lambda x:mean_particle_energy[x])

mean_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].mean()
df3_['mean_particle_mass'] = df3_['jet_id'].apply(lambda x:mean_particle_mass[x])

mean_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].mean()
df3_['mean_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:mean_particle_euclidean_distance[x])

mean_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].mean()
df3_['mean_particle_momentum'] = df3_['jet_id'].apply(lambda x:mean_particle_momentum[x])
##
mean_jet_px = df3_.groupby(['event_id'])['jet_px'].mean()
df3_['mean_jet_px'] = df3_['event_id'].apply(lambda x:mean_jet_px[x])

mean_jet_py = df3_.groupby(['event_id'])['jet_py'].mean()
df3_['mean_jet_py'] = df3_['event_id'].apply(lambda x:mean_jet_py[x])

mean_jet_pz = df3_.groupby(['event_id'])['jet_pz'].mean()
df3_['mean_jet_pz'] = df3_['event_id'].apply(lambda x:mean_jet_pz[x])

mean_jet_energy = df3_.groupby(['event_id'])['jet_energy'].mean()
df3_['mean_jet_energy'] = df3_['event_id'].apply(lambda x:mean_jet_energy[x])

mean_jet_mass = df3_.groupby(['event_id'])['jet_mass'].mean()
df3_['mean_jet_mass'] = df3_['event_id'].apply(lambda x:mean_jet_mass[x])

mean_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].mean()
df3_['mean_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:mean_jet_euclidean_distance[x])

##最大值
max_particle_px = df4_.groupby(['jet_id'])['particle_px'].max()
df3_['max_particle_px'] = df3_['jet_id'].apply(lambda x:max_particle_px[x])

max_particle_py = df4_.groupby(['jet_id'])['particle_py'].max()
df3_['max_particle_py'] = df3_['jet_id'].apply(lambda x:max_particle_py[x])

max_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].max()
df3_['max_particle_pz'] = df3_['jet_id'].apply(lambda x:max_particle_pz[x])

max_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].max()
df3_['max_particle_energy'] = df3_['jet_id'].apply(lambda x:max_particle_energy[x])

max_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].max()
df3_['max_particle_mass'] = df3_['jet_id'].apply(lambda x:max_particle_mass[x])

max_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].max()
df3_['max_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:max_particle_euclidean_distance[x])

max_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].max()
df3_['max_particle_momentum'] = df3_['jet_id'].apply(lambda x:max_particle_momentum[x])
##
max_jet_px = df3_.groupby(['event_id'])['jet_px'].max()
df3_['max_jet_px'] = df3_['event_id'].apply(lambda x:max_jet_px[x])

max_jet_py = df3_.groupby(['event_id'])['jet_py'].max()
df3_['max_jet_py'] = df3_['event_id'].apply(lambda x:max_jet_py[x])

max_jet_pz = df3_.groupby(['event_id'])['jet_pz'].max()
df3_['max_jet_pz'] = df3_['event_id'].apply(lambda x:max_jet_pz[x])

max_jet_energy = df3_.groupby(['event_id'])['jet_energy'].max()
df3_['max_jet_energy'] = df3_['event_id'].apply(lambda x:mean_jet_energy[x])

max_jet_mass = df3_.groupby(['event_id'])['jet_mass'].max()
df3_['max_jet_mass'] = df3_['event_id'].apply(lambda x:mean_jet_mass[x])

max_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].max()
df3_['max_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:max_jet_euclidean_distance[x])

##最小值
min_particle_px = df4_.groupby(['jet_id'])['particle_px'].min()
df3_['min_particle_px'] = df3_['jet_id'].apply(lambda x:min_particle_px[x])

min_particle_py = df4_.groupby(['jet_id'])['particle_py'].min()
df3_['min_particle_py'] = df3_['jet_id'].apply(lambda x:min_particle_py[x])

min_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].min()
df3_['min_particle_pz'] = df3_['jet_id'].apply(lambda x:min_particle_pz[x])

min_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].min()
df3_['min_particle_energy'] = df3_['jet_id'].apply(lambda x:min_particle_energy[x])

min_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].min()
df3_['min_particle_mass'] = df3_['jet_id'].apply(lambda x:min_particle_mass[x])

min_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].min()
df3_['min_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:min_particle_euclidean_distance[x])

min_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].min()
df3_['min_particle_momentum'] = df3_['jet_id'].apply(lambda x:min_particle_momentum[x])
##
min_jet_px = df3_.groupby(['event_id'])['jet_px'].min()
df3_['min_jet_px'] = df3_['event_id'].apply(lambda x:min_jet_px[x])

min_jet_py = df3_.groupby(['event_id'])['jet_py'].min()
df3_['min_jet_py'] = df3_['event_id'].apply(lambda x:min_jet_py[x])

min_jet_pz = df3_.groupby(['event_id'])['jet_pz'].min()
df3_['min_jet_pz'] = df3_['event_id'].apply(lambda x:min_jet_pz[x])

min_jet_energy = df3_.groupby(['event_id'])['jet_energy'].min()
df3_['min_jet_energy'] = df3_['event_id'].apply(lambda x:min_jet_energy[x])

min_jet_mass = df3_.groupby(['event_id'])['jet_mass'].min()
df3_['min_jet_mass'] = df3_['event_id'].apply(lambda x:min_jet_mass[x])

min_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].min()
df3_['min_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:min_jet_euclidean_distance[x])

##中位数
median_particle_px = df4_.groupby(['jet_id'])['particle_px'].median()
df3_['median_particle_px'] = df3_['jet_id'].apply(lambda x:median_particle_px[x])

median_particle_py = df4_.groupby(['jet_id'])['particle_py'].median()
df3_['median_particle_py'] = df3_['jet_id'].apply(lambda x:median_particle_py[x])

median_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].median()
df3_['median_particle_pz'] = df3_['jet_id'].apply(lambda x:median_particle_pz[x])

median_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].median()
df3_['median_particle_energy'] = df3_['jet_id'].apply(lambda x:median_particle_energy[x])

median_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].median()
df3_['median_particle_mass'] = df3_['jet_id'].apply(lambda x:median_particle_mass[x])

median_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].median()
df3_['median_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:median_particle_euclidean_distance[x])

median_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].median()
df3_['median_particle_momentum'] = df3_['jet_id'].apply(lambda x:median_particle_momentum[x])
##
median_jet_px = df3_.groupby(['event_id'])['jet_px'].median()
df3_['median_jet_px'] = df3_['event_id'].apply(lambda x:median_jet_px[x])

median_jet_py = df3_.groupby(['event_id'])['jet_py'].median()
df3_['median_jet_py'] = df3_['event_id'].apply(lambda x:median_jet_py[x])

median_jet_pz = df3_.groupby(['event_id'])['jet_pz'].median()
df3_['median_jet_pz'] = df3_['event_id'].apply(lambda x:median_jet_pz[x])

median_jet_energy = df3_.groupby(['event_id'])['jet_energy'].median()
df3_['median_jet_energy'] = df3_['event_id'].apply(lambda x:median_jet_energy[x])

median_jet_mass = df3_.groupby(['event_id'])['jet_mass'].median()
df3_['median_jet_mass'] = df3_['event_id'].apply(lambda x:median_jet_mass[x])

median_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].median()
df3_['median_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:median_jet_euclidean_distance[x])

##求和
sum_particle_px = df4_.groupby(['jet_id'])['particle_px'].sum()
df3_['sum_particle_px'] = df3_['jet_id'].apply(lambda x:sum_particle_px[x])

sum_particle_py = df4_.groupby(['jet_id'])['particle_py'].sum()
df3_['sum_particle_py'] = df3_['jet_id'].apply(lambda x:sum_particle_py[x])

sum_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].sum()
df3_['sum_particle_pz'] = df3_['jet_id'].apply(lambda x:sum_particle_pz[x])

sum_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].sum()
df3_['sum_particle_energy'] = df3_['jet_id'].apply(lambda x:sum_particle_energy[x])

sum_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].sum()
df3_['sum_particle_mass'] = df3_['jet_id'].apply(lambda x:sum_particle_mass[x])

sum_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].sum()
df3_['sum_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:sum_particle_euclidean_distance[x])

sum_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].sum()
df3_['sum_particle_momentum'] = df3_['jet_id'].apply(lambda x:sum_particle_momentum[x])
###
sum_jet_px = df3_.groupby(['event_id'])['jet_px'].sum()
df3_['sum_jet_px'] = df3_['event_id'].apply(lambda x:sum_jet_px[x])

sum_jet_py = df3_.groupby(['event_id'])['jet_py'].sum()
df3_['sum_jet_py'] = df3_['event_id'].apply(lambda x:sum_jet_py[x])

sum_jet_pz = df3_.groupby(['event_id'])['jet_pz'].sum()
df3_['sum_jet_pz'] = df3_['event_id'].apply(lambda x:sum_jet_pz[x])

sum_jet_energy = df3_.groupby(['event_id'])['jet_energy'].sum()
df3_['sum_jet_energy'] = df3_['event_id'].apply(lambda x:sum_jet_energy[x])

sum_jet_mass = df3_.groupby(['event_id'])['jet_mass'].sum()
df3_['sum_jet_mass'] = df3_['event_id'].apply(lambda x:sum_jet_mass[x])

sum_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].sum()
df3_['sum_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:sum_jet_euclidean_distance[x])

##标准差
std_particle_px = df4_.groupby(['jet_id'])['particle_px'].std()
df3_['std_particle_px'] = df3_['jet_id'].apply(lambda x:std_particle_px[x])

std_particle_py = df4_.groupby(['jet_id'])['particle_py'].std()
df3_['std_particle_py'] = df3_['jet_id'].apply(lambda x:std_particle_py[x])

std_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].std()
df3_['std_particle_pz'] = df3_['jet_id'].apply(lambda x:std_particle_pz[x])

std_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].std()
df3_['std_particle_energy'] = df3_['jet_id'].apply(lambda x:std_particle_energy[x])

std_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].std()
df3_['std_particle_mass'] = df3_['jet_id'].apply(lambda x:std_particle_mass[x])

std_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].std()
df3_['std_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:std_particle_euclidean_distance[x])

std_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].std()
df3_['std_particle_momentum'] = df3_['jet_id'].apply(lambda x:std_particle_momentum[x])
##
std_jet_px = df3_.groupby(['event_id'])['jet_px'].std()
df3_['std_jet_px'] = df3_['event_id'].apply(lambda x:std_jet_px[x])

std_jet_py = df3_.groupby(['event_id'])['jet_py'].std()
df3_['std_jet_py'] = df3_['event_id'].apply(lambda x:std_jet_py[x])

std_jet_pz = df3_.groupby(['event_id'])['jet_pz'].std()
df3_['std_jet_pz'] = df3_['event_id'].apply(lambda x:std_jet_pz[x])

std_jet_energy = df3_.groupby(['event_id'])['jet_energy'].std()
df3_['std_jet_energy'] = df3_['event_id'].apply(lambda x:std_jet_energy[x])

std_jet_mass = df3_.groupby(['event_id'])['jet_mass'].std()
df3_['std_jet_mass'] = df3_['event_id'].apply(lambda x:std_jet_mass[x])

std_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].std()
df3_['std_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:std_jet_euclidean_distance[x])

##方差
var_particle_px = df4_.groupby(['jet_id'])['particle_px'].var()
df3_['var_particle_px'] = df3_['jet_id'].apply(lambda x:var_particle_px[x])

var_particle_py = df4_.groupby(['jet_id'])['particle_py'].var()
df3_['var_particle_py'] = df3_['jet_id'].apply(lambda x:var_particle_py[x])

var_particle_pz = df4_.groupby(['jet_id'])['particle_pz'].var()
df3_['var_particle_pz'] = df3_['jet_id'].apply(lambda x:var_particle_pz[x])

var_particle_energy = df4_.groupby(['jet_id'])['particle_energy'].var()
df3_['var_particle_energy'] = df3_['jet_id'].apply(lambda x:var_particle_energy[x])

var_particle_mass = df4_.groupby(['jet_id'])['particle_mass'].var()
df3_['var_particle_mass'] = df3_['jet_id'].apply(lambda x:var_particle_mass[x])

var_particle_euclidean_distance = df4_.groupby(['jet_id'])['particle_euclidean_distance'].var()
df3_['var_particle_euclidean_distance'] = df3_['jet_id'].apply(lambda x:var_particle_euclidean_distance[x])

var_particle_momentum = df4_.groupby(['jet_id'])['particle_momentum'].var()
df3_['var_particle_momentum'] = df3_['jet_id'].apply(lambda x:var_particle_momentum[x])
##
var_jet_px = df3_.groupby(['event_id'])['jet_px'].var()
df3_['var_jet_px'] = df3_['event_id'].apply(lambda x:var_jet_px[x])

var_jet_py = df3_.groupby(['event_id'])['jet_py'].var()
df3_['var_jet_py'] = df3_['event_id'].apply(lambda x:var_jet_py[x])

var_jet_pz = df3_.groupby(['event_id'])['jet_pz'].var()
df3_['var_jet_pz'] = df3_['event_id'].apply(lambda x:var_jet_pz[x])

var_jet_energy = df3_.groupby(['event_id'])['jet_energy'].var()
df3_['var_jet_energy'] = df3_['event_id'].apply(lambda x:var_jet_energy[x])

var_jet_mass = df3_.groupby(['event_id'])['jet_mass'].var()
df3_['var_jet_mass'] = df3_['event_id'].apply(lambda x:var_jet_mass[x])

var_jet_euclidean_distance = df3_.groupby(['event_id'])['jet_euclidean_distance'].var()
df3_['var_jet_euclidean_distance'] = df3_['event_id'].apply(lambda x:var_jet_euclidean_distance[x])

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
df3.to_csv("train_jet.csv",index = False)
df4.to_csv("train_particle.csv",index = False)
df3_.to_csv("test_jet.csv",index = False)
df4_.to_csv("test_particle.csv",index = False)






