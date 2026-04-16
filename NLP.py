# -*- coding: utf-8 -*-
"""JenanTaibah_NLP.ipynb

# Natural Language Processing 

For this problem set, you'll also need the file conll2003train.txt --> if youre using colab
"""

!pip install gensim

"""# Part I.  Named Entity Recognition 
"""

# CONLL (Computational Natural Language Learning) 2003
# data from:
# https://data.deepai.org/conll2003.zip
# description of data:
# https://huggingface.co/datasets/eriktks/conll2003

from google.colab import files
uploaded = files.upload()

!ls

import csv

def load_ner_data(filename):
  lines = []
  with open(filename, mode='r') as myfile:
    spacereader = csv.reader(myfile, delimiter=' ')
    working_sentence = []
    working_ner_tags = []
    for row in spacereader:
      if len(row) == 0:
        if len(working_sentence) > 0:
          lines.append((working_sentence, working_ner_tags))
          working_sentence = []
          working_ner_tags = []
      elif len(row) == 4:
        if row[0] != '-DOCSTART-':
          working_sentence.append(row[0])
          working_ner_tags.append(process_ner_tag(row[3]))
  return lines

def process_ner_tag(tag):
  if tag == 'O':
    return 0
  tag = tag[2:] 
  tag_dict = {
      'PER': 1,
      'ORG': 2,
      'LOC': 3,
      'MISC': 4
  }
  return tag_dict[tag]


all_tuples = load_ner_data('conll2003train.txt')

len(all_tuples)

print(all_tuples[0])


MAX_SENTENCES = 1000
tuples = all_tuples[:MAX_SENTENCES]


import gensim.downloader as api

wv = api.load('word2vec-google-news-300')

import numpy as np
def words_to_word2vec_matrix(tuple_list, wv):
    feature_rows = []
    label_list = []

    for words, word_labels in tuple_list:
        for word, label in zip(words, word_labels):
            if word in wv:
                vec = wv[word]
            else:
                vec = np.zeros(300)
            feature_rows.append(vec)
            label_list.append(label)

    features = np.array(feature_rows)
    labels = np.array(label_list)
    return features, labels

features, labels = words_to_word2vec_matrix([(['Sonic', 'is', 'fast'], [1, 0, 0])], wv)
print(features.shape) # expect (3, 300)
print(labels.shape) # expect (3,)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

features, labels = words_to_word2vec_matrix(tuples, wv)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=340
)

rf = RandomForestClassifier(n_estimators=200, random_state=340)
rf.fit(X_train, y_train)

print(f"Accuracy  {rf.score(X_test, y_test):.4f}")  # expect ~0.94


from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, rf.predict(X_test)
)

class_names = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}

print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F-score':>10} {'Support':>10}")
print("-" * 52)
for i, (p, r, f, s) in enumerate(zip(precision, recall, fscore, support)):
    print(f"{class_names.get(i, str(i)):<8} {p:>10.4f} {r:>10.4f} {f:>10.4f} {s:>10}")


import numpy as np

sentence1 = ['Turkey', 'closed', 'its', 'borders', 'today']
sentence2 = ['Turkey', 'is', 'a', 'Thanksgiving', 'tradition']

def sentence_to_vectors(sentence, wv):
    return [wv[word] if word in wv else np.zeros(300) for word in sentence]

vecs1 = sentence_to_vectors(sentence1, wv)
vecs2 = sentence_to_vectors(sentence2, wv)


turkey_vec = vecs1[0]

dots1 = []
for i, (word, vec) in enumerate(zip(sentence1, vecs1)):
    if i == 0:
        continue
    dot = np.dot(turkey_vec, vec)
    dots1.append((word, dot))
    print(f"Turkey · {word:15s} = {dot:.4f}")

best_word1 = max(dots1, key=lambda x: x[1])[0]
print(f"\nLargest dot product: '{best_word1}'")


dots2 = []
for i, (word, vec) in enumerate(zip(sentence2, vecs2)):
    if i == 0:
        continue
    dot = np.dot(turkey_vec, vec)
    dots2.append((word, dot))
    print(f"Turkey · {word:15s} = {dot:.4f}")

best_word2 = max(dots2, key=lambda x: x[1])[0]
print(f"\nLargest dot product: '{best_word2}'")

def softmax(values):
    v = np.array(values)
    v = v - v.max()
    exp_v = np.exp(v)
    return exp_v / exp_v.sum()

raw_dots1 = [d for (_, d) in dots1]
weights1 = softmax(raw_dots1)

for (word, _), w in zip(dots1, weights1):
    print(f"  {word:15s}: {w:.6f}")
print(f"Sum = {weights1.sum():.6f}")

raw_dots2 = [d for (_, d) in dots2]
weights2 = softmax(raw_dots2)

for (word, _), w in zip(dots2, weights2):
    print(f"  {word:15s}: {w:.6f}")
print(f"Sum = {weights2.sum():.6f}")


non_turkey_vecs1 = [vecs1[i] for i in range(len(sentence1)) if i != 0]

attention_vec1 = sum(w * vec for w, vec in zip(weights1, non_turkey_vecs1))
print("Sentence 1 attention vector (first 5 dims):", attention_vec1[:5])

non_turkey_vecs2 = [vecs2[i] for i in range(len(sentence2)) if i != 0]

attention_vec2 = sum(w * vec for w, vec in zip(weights2, non_turkey_vecs2))
print("Sentence 2 attention vector (first 5 dims):", attention_vec2[:5])


class_names_full = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}
turkey_vec = vecs1[0]

for weight in np.arange(0.5, 100.0, 0.5):
    pred1 = rf.predict((turkey_vec + weight * attention_vec1).reshape(1, -1))[0]
    pred2 = rf.predict((turkey_vec + weight * attention_vec2).reshape(1, -1))[0]

    if pred1 == 3 and pred2 != 3:
        print(f"WEIGHT = {weight}")
        print(f"  Sentence 1 Turkey → {class_names_full[pred1]}")
        print(f"  Sentence 2 Turkey → {class_names_full[pred2]}")
        break


!pip install transformers datasets evaluate seqeval


from datasets import load_dataset

raw_dataset = load_dataset("siddharthtumre/jnlpba-split")

"""Let's take a look at what a HuggingFace dataset looks like:"""

raw_dataset

raw_dataset['train'][0]

# Some BERT code adapted from
# https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb

import transformers as ppb

WEIGHTS = 'distilbert-base-uncased'
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

tokenizer = get_tokenizer()



def get_tokens(word, tokenizer):
    token_list = tokenizer.encode(word)
    return token_list[1:-1] 


import numpy as np

def dataset_to_bert_input_and_labels(dataset, tokenizer, max_sentences):
    CLS_ID, SEP_ID = 101, 102

    token_rows, label_rows = [], []
    n = min(max_sentences, len(dataset))

    for idx in range(n):
        example = dataset[idx]
        words, ner_tags = example['tokens'], example['ner_tags']

        row_tokens = [CLS_ID]
        row_labels = [0]  

        for word, tag in zip(words, ner_tags):
            if tag % 2 == 1:
                tag = tag + 1

            sub_tokens = tokenizer.encode(word)[1:-1] 
            if len(sub_tokens) == 0:
                continue

            row_tokens.extend(sub_tokens)
            row_labels.extend([tag] * len(sub_tokens))

        row_tokens.append(SEP_ID)
        row_labels.append(0)  

        token_rows.append(row_tokens)
        label_rows.append(row_labels)

    max_len = max(len(row) for row in token_rows)

    padded_tokens = np.zeros((n, max_len), dtype=np.int64)
    padded_labels = []

    for i, (tok_row, lab_row) in enumerate(zip(token_rows, label_rows)):
        length = len(tok_row)
        padded_tokens[i, :length] = tok_row
        padded_labels.append(lab_row + [0] * (max_len - length))

    return padded_tokens, padded_labels

dataset_to_bert_input_and_labels(raw_dataset['train'], tokenizer, 2) 

import torch

def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

def get_bert_vectors(model, padded_tokens):
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    return word_vecs[0][:,:,:].numpy()

train_input, labels = dataset_to_bert_input_and_labels(raw_dataset['train'], tokenizer, 800)
model = get_model()
bert_result = get_bert_vectors(model, train_input)

print(bert_result.shape) # Expect (800, 180, 768)


import numpy as np

def labels_and_bert_to_sklearn(labels, bert_result):
    feature_list, label_list = [], []

    for i in range(bert_result.shape[0]):
        norms = np.linalg.norm(bert_result[i], axis=1)
        non_pad = np.where(norms > 1e-6)[0]

        if len(non_pad) == 0:
            continue
        word_positions = non_pad[1:-1]
        for j in word_positions:
            feature_list.append(bert_result[i, j, :])
            label_list.append(labels[i][j])

    return np.array(feature_list), np.array(label_list)

bert_features_train, bert_labels_train = labels_and_bert_to_sklearn(labels, bert_result)

"""III.4, 4 points) Call the following code block to get test data as well.  Then train a scikit-learn RandomForestClassifier with 200 estimators and random state 340 - this will take about 6 minutes on Colab - and evaluate the classifier on the test set.  You can expect an accuracy of about 80%."""

test_input, test_labels = dataset_to_bert_input_and_labels(raw_dataset['validation'], tokenizer, 100)
bert_result_test = get_bert_vectors(model, test_input)
bert_features_test, bert_labels_test = labels_and_bert_to_sklearn(test_labels, bert_result_test)

from sklearn.ensemble import RandomForestClassifier

bert_rf = RandomForestClassifier(n_estimators=200, random_state=340)
bert_rf.fit(bert_features_train, bert_labels_train)

print(f"BERT + RF accuracy: {bert_rf.score(bert_features_test, bert_labels_test):.4f}")

from transformers import pipeline

token_classifier = pipeline(
  "token-classification",
  "dbmdz/bert-large-cased-finetuned-conll03-english",
  aggregation_strategy="simple",
)

from datasets import load_dataset
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, pipeline

data = load_dataset("eriktks/conll2003", split="test", revision="refs/convert/parquet").shuffle(seed=340).select(range(1000))
task_evaluator = evaluator("token-classification")

eval_results = task_evaluator.compute(
    model_or_pipeline="dbmdz/bert-large-cased-finetuned-conll03-english",
    data=data,
    metric="seqeval"
)

print(eval_results)
