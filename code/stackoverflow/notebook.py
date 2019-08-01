# Deep Products: Deep Tag Labeler

## This is the first project for the book Deep Products, about using NLP and weakly supervised learning to build complete machine learning products. Using the non-code text of Stack Overflow posts (question and answers) to tag them using a multi-class, multi-label classifier using LSTMs and Emlo embeddings.

import os
import re

from keras import backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm_notebook

# Load 14 Million Answered Questions with Tags from Stack Overflow

## We load all answered questions from Stack Overflow. This data was converted from XML to JSON and then sampled using Spark on a single `r5.12xlarge` machine cluster with [code/stackoverflow/sample_json.spark.py](stackoverflow/sample_json.spark.py).

import glob
from multiprocessing import Pool

def read_parquet(path):
    return pd.read_parquet(
        path,
        columns=['_Body', '_Tags'],
        # filters=[('_Tags','!=',None),],
        engine='pyarrow'
    )

parquet_files = glob.glob("data/Questions.Stratified.5000.parquet/*.parquet")

pool = Pool(16)
posts_df = pd.concat(
    pool.map(read_parquet, parquet_files),
    ignore_index=True
)

posts_df.head(5)

l_values = [l for l in posts_df._Tags.values]
labels = np.array(l_values)
labels

'{:,}'.format(len(posts_df.index))

labels.sum(axis=0).shape[0]

## Try Different Thresholds for Filtering Tags by Frequency

## The higher the threshold, the fewer classes, the less sparse the data, the easier the learning task.

from collections import defaultdict

tag_counts = defaultdict(int)

for row in tag_posts['_Tag_List']:
    for tag in row:
        tag_counts[tag] += 1

for i in [0, 10, 20, 50, 100, 1000, 5000]:
    filtered_tags = list(filter(lambda x: x > i, tag_counts.values()))
    print('There are {:,} tags with more than {:,} count'.format(len(filtered_tags), i))

MIN_TAGS = 5000

record_count = len([i for i in filter(lambda x: x > MIN_TAGS, tag_counts.values())])
record_count

## Map from Tags to IDs

all_tags = set()
for row in tag_posts['_Tag_List']:
    for tag in row:
        if tag_counts[tag] > MIN_TAGS:
            all_tags.add(tag)

print('Total unique tags with {:,} occurrences: {:,}'.format(MIN_TAGS, len(all_tags)))
sorted_all_tags = sorted(all_tags)

tag_to_id = {val:i for i, val in enumerate(sorted_all_tags)}
id_to_tag = {i:val for i, val in enumerate(sorted_all_tags)}

## One Hot Encode Tag Lists

labels = []
tag_list = tag_posts['_Tag_List'].tolist()

# Loop through every post...
for i, tag_set in enumerate(tag_posts['_Tag_List'].tolist()):
    # Then build a record_count element wide list for each tag present
    label_row = []
    for tag in sorted_all_tags:
        if tag in tag_list[i]:
            label_row.append(1)
        else:
            label_row.append(0)
    labels.append(label_row)
    
tag_labels = [id_to_tag[key_id] for key_id in sorted(id_to_tag.keys()) if tag_counts[id_to_tag[key_id]] > MIN_TAGS]

len(labels), len(labels[0]), len(tag_labels)

## Extract/Tokenize Non-Code Text from Posts

## We leave posts' source code out for now because it will need a different embedding and thus multiple inputs.

from bs4 import BeautifulSoup


N_CORES = 12
MAX_LEN = 100
PAD_TOKEN = '__PAD__'
BATCH_SIZE = 64

def extract_text_series(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    tokens = doc.text.split()
    padded_tokens = [tokens[i] if len(tokens) > i else PAD_TOKEN for i in range(0, MAX_LEN)]
    return padded_tokens

def extract_text_df(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x['_Body'], 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    tokens = doc.text.split()
    padded_tokens = [tokens[i] if len(tokens) > i else PAD_TOKEN for i in range(0, MAX_LEN)]
    return padded_tokens

post_text = tag_posts.map_partitions(lambda df: df.apply(extract_text_df, axis=1)).reset_index(drop=True)

post_text.head(5)

len(post_text.index), len(post_text.iloc[0]), len(labels), len(labels[0])

# Validate the posts match the labels
assert(len(post_text.index) == len(labels))
print('We are left with {:,} example posts'.format(len(post_text.index)))

## Make Record Count a Multiple of the Batch Size and Post Sequence Length

## The Elmo embedding requires that the number of records be a multiple of the batch size times the number of tokens in the padded posts.

import math

# Filter label rows that don't have any positive labels
label_mx = np.array(labels)
max_per_row = label_mx.max(axis=1)
non_zero_index = np.nonzero(max_per_row)[0]

label_mx = label_mx[non_zero_index]

# Filter the posts to match
post_text = post_text[post_text.index.isin(non_zero_index)]
post_text = post_text.to

assert(post_text.shape[0] == label_mx.shape[0])
print('Unfiltered Counts: {:,} {:,}'.format(post_text.shape[0], label_mx.shape[0]))

# training_count must be a multiple of the BATCH_SIZE times the MAX_LEN for the Elmo embedding layer
highest_factor = math.floor(post_text.shape[0] / (BATCH_SIZE * MAX_LEN))
training_count = highest_factor * BATCH_SIZE * MAX_LEN
print('Highest Factor: {:,} Training Count: {:,}'.format(highest_factor, training_count))

label_mx = label_mx[0:training_count]
post_text = post_text[0:training_count]

assert(post_text.shape[0] == label_mx.shape[0])
print('Final Counts: {:,} {:,}'.format(post_text.shape[0], label_mx.shape[0]))

post_text

## Create an Elmo Embedding Layer using Tensorflow Hub

## Note that this layer takes a padded two-dimensional array of strings.

# From https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/

sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(BATCH_SIZE*[MAX_LEN])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

## Experimental Setup

## We `train_test_split` rather than k-fold cross validate because it is too expensive.

from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.15

X_train, X_test, y_train, y_test = train_test_split(
    post_text,
    label_mx,
    test_size=TEST_SPLIT,
    random_state=34
)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

## Create an LSTM Model to Classify Posts into Tags

## We use the padded/tokenized posts as input, an Elmo embedding feeding an Long-Short-Term-Memory (LSTM) layer followed by a Dense layer with the same number of output neurons as our tag list.

## We use focal loss as a loss function, which is used in appliations like object detection, because it 

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum(K.abs( (y_true - y_pred) * (K.log(y_true / y_pred))), axis=-1)

# from keras.layers import Input, concatenate, Activation, Dense, LSTM, BatchNormalization, Embedding, Dropout, Lambda, Bidirectional
# from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
# from keras.models import Model
# from keras.optimizers import Adam
# from keras_metrics import precision, f1_score, false_negative, true_positive, false_positive, true_negative

# # Text model
# text_input = Input(shape=(MAX_LEN,), dtype=tf.string)

# elmo_embedding = Lambda(ElmoEmbedding, output_shape=(MAX_LEN, 1024))(text_input)

# text_lstm = LSTM(
#     input_shape=(MAX_LEN, 1024,),
#     units=512,
#     recurrent_dropout=0.2,
#     dropout=0.2)(elmo_embedding)

# text_dense = Dense(200, activation='relu')(text_lstm)

# text_output = Dense(record_count, activation='sigmoid')(text_dense)

# text_model = Model(
#     inputs=text_input, 
#     outputs=text_output
# )



# from sklearn.metrics import hamming_loss

# from keras.optimizers import Adam
# adam = Adam(lr=0.0005)

# text_model.compile(
#     loss='binary_crossentropy',
#     optimizer=adam,
#     metrics=[
#         precision_m,
#         recall_m,
#         f1_m,
#         'mae',
#         abs_KL_div,
#         'accuracy'
#     ]
# )
# 
# text_model.summary()

## Compute Sample and Class Weights

## Because we have skewed classes and multiple classes per example, we employ sample or class weights which weight the importance of each row according to the relative frequency of their labels.

from sklearn.utils.class_weight import compute_sample_weight

train_sample_weights = compute_sample_weight('balanced', y_train)
test_sample_weights = compute_sample_weight('balanced', y_test)

class_weights = {}
for i, tag in enumerate(sorted_all_tags):
    class_weights[i] = label_mx.shape[0] / tag_counts[tag]

class_weights

## Establish a Log for Performance

## Simple Baseline Model using `Conv1D`



from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, GlobalMaxPool1D, Conv1D, Dense, Activation, Dropout, Lambda
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model
from keras_metrics import precision, f1_score, false_negative, true_positive, false_positive, true_negative


def build_model(max_len, label_count, dropout_ratio=0.1, filter_length=50):
    
    text_input = Input(shape=(max_len,), dtype=tf.string)

    elmo_embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(text_input)

    dropout = Dropout(dropout_ratio)(elmo_embedding)

    conv1d = Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1)(dropout)

    global_1d = GlobalMaxPool1D()(conv1d)

    dense = Dense(label_count, activation='sigmoid')(global_1d)

    text_model = Model(
        inputs=text_input, 
        outputs=dense
    )
    
    parallel_model = multi_gpu_model(text_model, gpus=2)

    parallel_model.compile(
        optimizer='adam',
        loss=abs_KL_div,#'binary_crossentropy',
        metrics=[
            'categorical_accuracy',
            precision_m,
            recall_m,
            f1_m,
            'mae',
            abs_KL_div,
            true_positive(),
            false_positive(),
            true_negative(),
            false_negative(),
            'accuracy',
        ]
    )
    text_model.summary()
    
    return text_model

text_model = build_model(MAX_LEN, y_train.shape[1])

callbacks = [
    ReduceLROnPlateau(), 
    EarlyStopping(patience=4), 
    # ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

history = text_model.fit(
    X_train, 
    y_train,
    class_weight=class_weights,
    epochs=20,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# from keras.callbacks import EarlyStopping

# EPOCHS = 4

# history = text_model.fit(
#     X_train,
#     y_train,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[
#         EarlyStopping(monitor='loss', patience=1, min_delta=0.0001),
#         EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001),
#     ],
#     class_weight=class_weights,
#     # sample_weight=train_sample_weights,
#     validation_data=(X_test, y_test)
# )

accr = text_model.evaluate(X_test, y_test, sample_weight=test_sample_weights)
[i for i in zip(accr, text_model.metrics_names)]

%matplotlib inline

import matplotlib.pyplot as plt

print(history.history)
# summarize history for accuracy
plt.plot(history.history['val_loss'])
plt.plot(history.history['f1_m'])
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['precision_m'])
plt.title('model accuracy')
plt.ylabel('metric')
plt.xlabel('epoch')
plt.legend(['val_loss', 'f1', 'categorical accuracy', 'MAE', 'precision'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import statistics

from sklearn.metrics import hamming_loss, jaccard_score
import keras.backend as K
import tensorflow as tf

y_pred = text_model.predict(X_test)

sess = tf.Session()
best_cutoff = 0
max_score = 0
with sess.as_default():
    for cutoff in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8]:
        y_pred_bin = K.greater(y_pred, cutoff).eval()
        print('Cutoff: {:,}'.format(cutoff))
        print('Hamming loss: {:,}'.format(
            hamming_loss(y_test, y_pred_bin)
        ))
        scores = []
        for j_type in ['micro', 'macro', 'weighted']:
            j_score = jaccard_score(y_test, y_pred_bin, average=j_type)
            print('Jaccard {} score: {:,}'.format(
                j_type,
                j_score
            ))
            scores.append(j_score)
        print('')
        mean_score = statistics.mean(scores)
        if mean_score > max_score:
            best_cutoff = cutoff
            max_score = mean_score

print('Best cutoff was: {:,} with mean jaccard score of {:,}'.format(best_cutoff, max_score))

y_pred

from sklearn.metrics import classification_report, multilabel_confusion_matrix

y_pred = text_model.predict(X_test, batch_size=32, verbose=1)
y_pred_bool = np.where(y_pred > best_cutoff, 1, 0)

print(classification_report(y_test, y_pred_bool))

print(multilabel_confusion_matrix(y_test, y_pred_bool))

## View the Results

## Now lets map from the one-hot-encoded tags back to the text tags and view them alongside the text of the original posts to sanity check the model and see if it really works.

predicted_tags = []
for test, pred in zip(y_test, y_pred_bool):
    tags = []
    for i, val in enumerate(test):
        if pred[i] == 1.0:
            tags.append(sorted_all_tags[i])
    predicted_tags.append(tags)

for text, tags in zip(X_test, predicted_tags):
    print(' '.join(text), tags)
