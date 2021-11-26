from mangoes.modeling import BERTForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocess_data import map_targets, accuracy, split_data_by_class
import pandas as pd
import os


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


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

model = BERTForSequenceClassification.load("dbmdz/bert-base-french-europeana-cased",
                                           "dbmdz/bert-base-french-europeana-cased", labels=list(set(train_classes)),
                                           )
model.train(train_text=train_text, train_targets=train_classes,
            eval_text=test_text, eval_targets=test_classes, evaluation_strategy="steps", eval_steps=10,
            output_dir="./testing/", max_len=512, num_train_epochs=3, compute_metrics=compute_metrics, freeze_base=True,
            )

train_pred = []
for i in range(len(train_text)):
    train_pred.append(model.predict(train_text[i])["label"])
print(accuracy(train_pred, train_classes))

test_pred = []
for i in range(len(test_text)):
    test_pred.append(model.predict(test_text[i])["label"])
print(accuracy(test_pred, test_classes))


