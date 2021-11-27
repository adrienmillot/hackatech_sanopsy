from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import *

from embeddings import *
from preprocess_data import map_targets, accuracy,fuzzy_accuracy


data_dir = "data/csv/"

manual_data_balancing = True

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



train_classes, test_classes, target_mapping = map_targets(train_classes, test_classes)


print(Counter(train_classes))
print(Counter(test_classes))

sub_keywords_map = get_keyword_dict(train_sub_keywords)
train_sparse_vectors = get_sparse_subkeyword_vectors(train_sub_keywords, sub_keywords_map)
test_sparse_vectors = get_sparse_subkeyword_vectors(test_sub_keywords, sub_keywords_map)

train_dense_vectors, test_dense_vectors = get_svd_dense_vectors(train_sparse_vectors, test_sparse_vectors,
                                                                dense_size=32, num_iter=10)
print(train_dense_vectors.shape)
ros = RandomOverSampler(random_state=0)
res_train_dense_vectors, res_train_classes = ros.fit_resample(train_dense_vectors, train_classes)


clf = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=200)
clf.fit(res_train_dense_vectors, res_train_classes)
pred_train = clf.predict(train_dense_vectors)
inv_target_mapping = {v: k for k, v in target_mapping.items()}
print(accuracy(pred_train, train_classes))
pred_train_str = [inv_target_mapping[k] for k in pred_train]
# print(fuzzy_accuracy(pred_train_str, train_lists_classes,2))


print("---")
for i in range(len(target_mapping)):
    print(f"{inv_target_mapping[i]}: {len([x for x in pred_train if x == i])}/ {len([x for x in train_classes if x == i])}")

pred_test = clf.predict(test_dense_vectors)
print(accuracy(pred_test, test_classes))
pred_test_str = [inv_target_mapping[k] for k in pred_test]
# print(fuzzy_accuracy(pred_test_str, test_lists_classes,2))
inv_target_mapping = {v: k for k, v in target_mapping.items()}

for i in range(len(target_mapping)):
    print(f"(pred/real):{inv_target_mapping[i]}: {len([x for x in pred_test if x == i])} / {len([x for x in test_classes if x == i])}")
print("break")