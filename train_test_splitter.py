#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 07:04:49 2018

Split CUB2002011 to test/train folders
@author: n-kamiya
"""
import os
import pandas as pd
from database import cub2002011_db

#%%create database
con , cursor  = cub2002011_db.create_cub_db()
#%%load data frames
df = pd.read_sql("select * from CUB2002011", con)
df_train_path = pd.read_sql("select * from CUB2002011 where train = 1", con)
df_test_path = pd.read_sql("select * from CUB2002011 where train = 0", con)
#%%

train_path = []
test_path = []

train_path=(df_train_path.iloc[:,3].values)
test_path=(df_test_path.iloc[:,3].values)


train_f = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/train/"
test_f = "/home/n-kamiya/datasets/CUB2002011/CUB_200_2011/test/"
#%% remove files

#for file in train_path:
#    os.remove(test_f+file)
#for file in test_path:
#    os.remove(train_f+file)