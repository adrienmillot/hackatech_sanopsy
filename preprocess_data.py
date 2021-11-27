import pandas as pd
import os
import numpy as np
import random


def map_targets(train_targets, test_targets):
    target_mapping = {t: i for i, t in enumerate(list(set(train_targets)))}
    missing = [x for x in set(test_targets) if x not in train_targets]
    for key in missing:
        target_mapping[key] = max(target_mapping.values())+1
    train_targets = [target_mapping[x] for x in train_targets]
    test_targets = [target_mapping[x] for x in test_targets]
    return train_targets, test_targets, target_mapping


def accuracy(predicted, targets):
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == targets[i]:
            correct += 1
    return float(correct) / float(len(predicted))


def fuzzy_accuracy(predicted, targets_list, max_place):    
    correct = 0
    for i in range(len(predicted)):
        for j in range(0,max_place):
            if(len(targets_list[i].split(';'))<j+1):
                break         
            if predicted[i] == targets_list[i].split(';')[j]:
                correct += 1
                break
    return float(correct) / float(len(predicted))


def split_data_by_class(ratio, classes):
    """
    :param ratio: ratio of example to put in test set
    :param classes: list of classes, one for each example
    :return: train indices, test indices
    """
    train_indices = []
    test_indices = []
    for label in set(classes):
        label_indices = [x[0] for x in enumerate(classes) if x[1] == label]
        random.shuffle(label_indices)
        test_cutoff = max(int(len(label_indices) * ratio), 1)
        test_indices += label_indices[:test_cutoff]
        train_indices += label_indices[test_cutoff:]
    return train_indices, test_indices


def load_labeled_data(data_path):
    """
    :param data_path: path to 'dataset_trié.csv'
    :return: dataframe with NA values dropped
    """
    data = pd.read_csv(data_path, header=None).iloc[:, [1, 2, 3]]
    data = data.dropna()
    return data


def load_keyword_class_dict(data_path):
    """
    :param data_path: path to 'therapies_groupes_mots-clés.csv'
    :return:
    dict:
        keys: keywords
        values: list of classes associated with this keyword
    """
    df_keys = pd.read_csv(data_path)
    for i in range(len(df_keys)):
        df_keys["Courants"][i] = df_keys["Courants"][i].strip()
    courants = set(df_keys["Courants"])
    class_words = {}
    for courant in courants:
        l = np.array(df_keys[df_keys["Courants"] == courant].iloc[:, 1:]).reshape(-1)
        l = l[pd.isnull(l) != True]
        l = [i.strip('[]') for i in l]
        class_words[courant] = l
    clean_keywords = {}  # keys: keywords, values: classes associated with that keyword
    for k in class_words.keys():
        for v in class_words[k]:
            if ' '.join(v.lower().split()) in clean_keywords:
                clean_keywords[' '.join(v.lower().split())].append(' '.join(k.lower().split()))
            else:
                clean_keywords[' '.join(v.lower().split())] = [' '.join(k.lower().split())]
    return clean_keywords


def get_labels_from_keywords(keywords, keyword_class_dict):
    """
    given a list of keywords for each example and a dict mapping keywords to associated class, return the class most
    associated with each example based on it's keywords.
    :param keywords: list of keywords for each example
    :return: list of labels, one for each example
    """
    labels = []
    main_labels = []
    for current_keywords in keywords:
        current_labels = {}
        found_keywords = []
        for keyword in keyword_class_dict.keys():  # iterate over all keywords
            if keyword in current_keywords:  # if keyword in text, add labels associated with keyword
                found_keywords.append(keyword)
                for keyword_class in keyword_class_dict[keyword]:
                    
                    if keyword_class in current_labels:
                        current_labels[keyword_class] += 1
                    else:
                        current_labels[keyword_class] = 1
        current_labels = list(current_labels.items())
        current_labels.sort(key=lambda x: x[1], reverse=True)
        
        
        i=0
        label = current_labels[0][0]
        while len(current_labels) > i+1:
            i+=1
            label = ';'.join([label, current_labels[i][0]])
        
        labels.append(label)
        main_labels.append(current_labels[0][0])
    return main_labels,labels


if __name__ == "__main__":
    data_dir = "data/csv/"
    labeled_data = load_labeled_data(os.path.join(data_dir, "dataset_trié.csv"))
    texts = labeled_data.iloc[:, 0].tolist()
    sub_keywords = labeled_data.iloc[:, 1].tolist()
    keywords = labeled_data.iloc[:, 2].tolist()
    # normalize keywords
    sub_keywords = [[' '.join(x.lower().split()) for x in current_keywords.split(";")] for current_keywords in sub_keywords]
    keywords = [[' '.join(x.lower().split()) for x in current_keywords.split(";")] for current_keywords in keywords]

    keyword_class_dict = load_keyword_class_dict(os.path.join(data_dir, "therapies_groupes_mots-clés.csv"))
    main_targets,targets = get_labels_from_keywords(keywords, keyword_class_dict)

    assert len(texts) == len(sub_keywords) == len(keywords) == len(targets)
    print(texts[0])
    print(sub_keywords[0])
    print(keywords[0])
    print(targets[0])
    # change to better csv format
    sub_keywords = [";".join(x) for x in sub_keywords]
    keywords = [";".join(x) for x in keywords]

    dataset = pd.DataFrame(zip(texts, sub_keywords, keywords,main_targets, targets), columns=['text', 'sub_keywords', 'keywords', 'main_class','classes'])
    dataset.to_csv("data/csv/full_dataset.csv", index=False,)

    print(dataset["classes"].value_counts())
    print(len(dataset["classes"].value_counts()))

