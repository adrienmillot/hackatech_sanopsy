#!/usr/bin/python
import numpy as np 
import pandas as pd




df = pd.read_csv("data/csv/data_labeled.csv",
                 header = None ).iloc[:,:-1]
df_keys = pd.read_csv("data/csv/therapies_mots_clef.csv" )

for i in range(len(df_keys)):
    df_keys["Courants"][i] = df_keys["Courants"][i].strip()

courants = set(df_keys["Courants"])
class_words = {}

for courant in courants:
    l = np.array(df_keys[df_keys["Courants"] == courant].iloc[:,1:]).reshape(-1)
    l = l[ pd.isnull(l) != True]
    l = [i.strip('[]') for i in l]
    class_words[courant] = l
