# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:50:22 2020
提交记录
1 把在code5的基础上把particle_category的正负统一为绝对值（正负粒子？），但这是有害特征得分从0.689下降到0.6888
@author: Woody
"""
import pandas as pd
from lightgbm import LGBMClassifier as lgb
df3 = pd.read_csv("train_jet5.csv")
df4 = pd.read_csv("train_particle5.csv")
df3_ = pd.read_csv("test_jet5.csv")
df4_ = pd.read_csv("test_particle5.csv")

df4['particle_category'] = df4['particle_category'].apply(lambda x : abs(x))

dummies = pd.get_dummies(df4['particle_category'],prefix ='abs_particle_category' )
df4[dummies.columns] = dummies
particle_category = dummies.columns
i = df4.groupby(['jet_id'],as_index = False)[dummies.columns].sum()
i['jet_id'] = df4['jet_id'].unique()
df3 = pd.merge(df3,i,on = 'jet_id',how = 'left')

dummies = pd.get_dummies(df4_['particle_category'],prefix ='abs_particle_category' )
df4_[dummies.columns] = dummies
particle_category = dummies.columns
i = df4_.groupby(['jet_id'],as_index = False)[dummies.columns].sum()
i['jet_id'] = df4_['jet_id'].unique()
df3_ = pd.merge(df3_,i,on = 'jet_id',how = 'left')

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
df3.to_csv("train_jet6.csv",index = False)
df4.to_csv("train_particle6.csv",index = False)
df3_.to_csv("test_jet6.csv",index = False)
df4_.to_csv("test_particle6.csv",index = False)

    
    




