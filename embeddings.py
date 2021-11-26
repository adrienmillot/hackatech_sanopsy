import numpy as np
import pandas as pd
import os
from preprocess_data import split_data_by_class
from sklearn.decomposition import TruncatedSVD


def get_keyword_dict(train_sub_keywords):
    sub_keywords_map = {}
    for train_sub_keyword_list in train_sub_keywords:
        for sub_keyword in train_sub_keyword_list.split(";"):
            if sub_keyword not in sub_keywords_map:
                sub_keywords_map[sub_keyword] = len(sub_keywords_map)
    return sub_keywords_map


def get_sparse_subkeyword_vectors(sub_keywords, key_word_dict):
    """

    :param sub_keywords: list of subkeywords strings: strings of subkeywords delimited by ";"
    :param key_word_dict:
    :return:
    """
    keyword_counts = np.zeros((len(sub_keywords), len(key_word_dict)))
    for i in range(len(sub_keywords)):
        for sub_keyword in sub_keywords[i].split(";"):
            if sub_keyword in key_word_dict:
                keyword_counts[i, key_word_dict[sub_keyword]] += 1
    return keyword_counts


def get_svd_dense_vectors(train_sparse_vectors, test_sparse_vectors, dense_size=32, num_iter=10):
    svd = TruncatedSVD(n_components=dense_size, n_iter=num_iter, random_state=42)
    svd.fit(train_sparse_vectors)
    train_dense_vectors = svd.transform(train_sparse_vectors)
    test_dense_vectors = svd.transform(test_sparse_vectors)
    return train_dense_vectors, test_dense_vectors


if __name__ == "__main__":
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

    sub_keywords_map = get_keyword_dict(train_sub_keywords)
    train_sparse_vectors = get_sparse_subkeyword_vectors(train_sub_keywords, sub_keywords_map)
    test_sparse_vectors = get_sparse_subkeyword_vectors(test_sub_keywords, sub_keywords_map)

    print(train_sparse_vectors.shape)

    train_dense_vectors, test_dense_vectors = get_svd_dense_vectors(train_sparse_vectors, test_sparse_vectors,
                                                                    dense_size=32, num_iter=10)
    print(train_dense_vectors.shape)
