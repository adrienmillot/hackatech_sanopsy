import pandas as pd
import os
import numpy as np
import random


from preprocess_data import split_data_by_class


if __name__ == "__main__":
    data_dir = "data/csv/"
    full_data_path = os.path.join(data_dir, "full_dataset.csv")

    preprocessed_data = pd.read_csv(full_data_path, sep=',')
    
    
    classes = preprocessed_data.iloc[:, 3].tolist()
    
    
    
    train_indices, test_indices = split_data_by_class(0.2, classes)
    train_data = preprocessed_data.iloc[train_indices]
    test_data = preprocessed_data.iloc[test_indices]
    
    
    print(test_data["main_class"].value_counts())
    print(train_data["main_class"].value_counts())
    
    
    

    
    
#     for i in preprocessed_data:
#         print(preprocessed_data[i])

    #ATTENTION SELON LE PREPROCESS, LES RATIOS SONT A CHANGER, ET L'ENTRAINEMENT PEUT AVOIR DES RÉSULTATS TRÈS DIFFÉRENTS
    train_data = train_data.drop(train_data[train_data['main_class']=="thérapie cognitivo-comportementale"].sample(frac=.70).index)
    train_data = train_data.drop(train_data[train_data['main_class']=="thérapie de soutien"].sample(frac=.20).index)
    
    train_data = train_data.append([train_data[train_data['main_class']=="thérapie systémique"]]*1)
    train_data = train_data.append([train_data[train_data['main_class']=="psychanalyse"]]*3)
    train_data = train_data.append([train_data[train_data['main_class']=="neuropsychologie"]]*4)
    train_data = train_data.append([train_data[train_data['main_class']=="sexothérapie"]]*14)
    train_data = train_data.append([train_data[train_data['main_class']=="psychiatrie"]]*3)
    train_data = train_data.append([train_data[train_data['main_class']=="approche centrée sur la personne"]]*1)
    train_data = train_data.append([train_data[train_data['main_class']=="thérapie de la gestalt"]]*4)
    train_data = train_data.append([train_data[train_data['main_class']=="thérapie psychocorporelle"]]*5)
    train_data = train_data.append([train_data[train_data['main_class']=="thérapies existentielles"]]*30)

    

    print(train_data["main_class"].value_counts())
    
    

    preprocessed_data.to_csv("data/csv/new_full_dataset.csv", index=False,)
    train_data.to_csv("data/csv/new_full_dataset_train.csv", index=False,)
    test_data.to_csv("data/csv/new_full_dataset_test.csv", index=False,)