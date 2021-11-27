import pandas as pd
import os

from collections import *

from preprocess_data import split_data_by_class, get_labels_from_keywords, accuracy,get_labels_from_keywords_with_ratio


data_dir = "data/csv/"

manual_data_balancing = False

if(not manual_data_balancing):

#     data = pd.read_csv(os.path.join(data_dir, "new_full_dataset.csv"))
    data = pd.read_csv(os.path.join(data_dir, "full_dataset.csv"))
     
     
     
    all_text = data.iloc[:, 0].tolist()
    sub_keywords = data.iloc[:, 1].tolist()
    keywords = data.iloc[:, 2].tolist()
    classes = data.iloc[:, 3].tolist()
    lists_classes = data.iloc[:, 4].tolist()
     
    train_indices, test_indices = split_data_by_class(0.2, classes)
    print(len(train_indices))
    print(len(test_indices))
     
    train_text = [all_text[i] for i in train_indices]
    train_sub_keywords = [sub_keywords[i] for i in train_indices]
    train_keywords = [keywords[i] for i in train_indices]
    train_classes = [classes[i] for i in train_indices]
    train_lists_classes = [lists_classes[i] for i in train_indices]
     
    test_text = [all_text[i] for i in test_indices]
    test_sub_keywords = [sub_keywords[i] for i in test_indices]
    test_keywords = [keywords[i] for i in test_indices]
    test_classes = [classes[i] for i in test_indices]
    test_lists_classes = [lists_classes[i] for i in test_indices]


    
    print(Counter(train_classes))
    print(Counter(test_classes))
    
else:
    
    ####################"""""""""
    data_train = pd.read_csv(os.path.join(data_dir, "new_full_dataset_train.csv"))
    data_test = pd.read_csv(os.path.join(data_dir, "new_full_dataset_test.csv"))
    
    
    
    train_text = data_train.iloc[:, 0].tolist()
    train_sub_keywords = data_train.iloc[:, 1].tolist()
    train_keywords = data_train.iloc[:, 2].tolist()
    train_classes = data_train.iloc[:, 3].tolist()
    train_lists_classes = data_train.iloc[:, 4].tolist()
    
    test_text = data_test.iloc[:, 0].tolist()
    test_sub_keywords = data_test.iloc[:, 1].tolist()
    test_keywords = data_test.iloc[:, 2].tolist()
    test_classes = data_test.iloc[:, 3].tolist()
    test_lists_classes = data_test.iloc[:, 4].tolist()
    
    ################"""







# build sub keyword -> associated class mapping
sub_keyword_to_class = {}
for i in range(len(train_text)):
    current_sub_keywords = train_sub_keywords[i].split(";")
    for keyword in current_sub_keywords:
        if keyword not in sub_keyword_to_class:
            sub_keyword_to_class[keyword] = []
        if train_classes[i] not in sub_keyword_to_class[keyword]:
            sub_keyword_to_class[keyword].append(train_classes[i])

class_keyword_counts = {}
for _, v in sub_keyword_to_class.items():
    for keyword_class in v:
        if keyword_class in class_keyword_counts:
            class_keyword_counts[keyword_class] += 1
        else:
            class_keyword_counts[keyword_class] = 1

found_sub_keywords = []
for i in range(len(train_text)):
    current_text = train_text[i].lower().replace("(", "").replace(")", "")
    current_sub_keywords = []
    for sub_keyword in sub_keyword_to_class.keys():
        if sub_keyword in current_text:
            current_sub_keywords.append(sub_keyword)
    if len(current_sub_keywords) == 0:
        print(f"Found no keywords {i}")
    found_sub_keywords.append(current_sub_keywords)

for lab in class_keyword_counts:
    print(f"{lab}: {len([x for x in train_classes if x == lab])}")
print("\n")
# train_predicted_classes,_ = get_labels_from_keywords(found_sub_keywords, sub_keyword_to_class)
train_predicted_classes,_ = get_labels_from_keywords_with_ratio(found_sub_keywords, sub_keyword_to_class)
for lab in class_keyword_counts:
    print(f"{lab}: {len([x for x in train_predicted_classes if x == lab])}")

print(train_predicted_classes)
print(accuracy(train_predicted_classes, train_classes))

found_sub_keywords = []
for i in range(len(test_text)):
    current_text = test_text[i].lower().replace("(", "").replace(")", "")
    current_sub_keywords = []
    for sub_keyword in sub_keyword_to_class.keys():
        if sub_keyword in current_text:
            current_sub_keywords.append(sub_keyword)
    if len(current_sub_keywords) == 0:
        print(f"Found no keywords {i}")
    found_sub_keywords.append(current_sub_keywords)

test_predicted_classes,_ = get_labels_from_keywords(found_sub_keywords, sub_keyword_to_class)
print(test_predicted_classes)
print(accuracy(test_predicted_classes, test_classes))

for lab in class_keyword_counts:
    print(f"{lab}: {len([x for x in test_predicted_classes if x == lab])}")

