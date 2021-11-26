import pandas as pd
import os
import numpy as np
from preprocess_data import split_data_by_class, get_labels_from_keywords, accuracy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data_dir = "data/csv/"
data = pd.read_csv(os.path.join(data_dir, "full_dataset.csv"))

all_text = data.iloc[:, 0].tolist()
sub_keywords = data.iloc[:, 1].tolist()
keywords = data.iloc[:, 2].tolist()
classes = data.iloc[:, 3].tolist()

train_indices, test_indices = split_data_by_class(0.2, classes)
print(len(train_indices))
print(len(test_indices))

train_text = [all_text[i] for i in train_indices]
train_sub_keywords = [sub_keywords[i] for i in train_indices]
train_keywords = [keywords[i] for i in train_indices]
train_classes = [classes[i] for i in train_indices]

test_text = [all_text[i] for i in test_indices]
test_sub_keywords = [sub_keywords[i] for i in test_indices]
test_keywords = [keywords[i] for i in test_indices]
test_classes = [classes[i] for i in test_indices]

# build sub keyword -> associated class mapping
sub_keyword_to_class = {}
all_train_sub_keywords = []
for i in range(len(train_text)):
    current_sub_keywords = train_sub_keywords[i].split(";")
    all_train_sub_keywords += current_sub_keywords
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

train_predicted_classes = get_labels_from_keywords(found_sub_keywords, sub_keyword_to_class)
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

test_predicted_classes = get_labels_from_keywords(found_sub_keywords, sub_keyword_to_class)
print(test_predicted_classes)
print(accuracy(test_predicted_classes, test_classes))
unique_subkeywords = np.unique(all_train_sub_keywords)[1:]

X = []
for txt in data.iloc[:,0]:
    X.append(np.char.count(txt,unique_subkeywords))

y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y)

clf = RandomForestClassifier().fit(X_train,y_train)

clf.predict(X_test)
score = clf.score(X_test, y_test)


    



