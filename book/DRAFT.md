# Weakly Supervised Learning: Do More with Less Data
##  Table of Contents

1. Chapter 1: Introduction
	1. Statistical Natural Language Processing
	2. Snorkel
	3. Transfer Learning
	4. Weak Supervision
	5. Distant Supervision
	6. Semi-Supervised Learning
	7. Looking Ahead
2. Collecting and Processing Data
	1. Introduction to Stack Overflow
	2. Collecting Stack Overflow Data
	3. Scaling ETL with PySpark and Elastic MapReduce
3. Transfer Learning
	1. Introduction to Transfer Learning
	2. Incorporating Elmo Embeddings
	3. Re-training Elmo Embeddings
	4. Other Embeddings
4. Weak Supervision
	1. Introduction to Weak Supervision
	2. Exploratory Data Analysis
	3. Data Programming with Snorkel
	4. Building New Labels
	5. Training the Final Model
5. Semi-Supervised Learning
	1. Introduction to Semi-Supervised Learning
	2. Creating a Model using Limited Data
	3. Classifying Unlabeled Data
	4. Improving the Model
6. Distant Supervision
	1. Introduction to Distant Supervision
	2. Extracting Positive Examples from Freebase
	3. Generating Negative Examples
	4. Training a Distantly Supervised Model
7. Model Management
	1. Introduction to Model Management
	2. Introducing <Model Management Framework>
	3. Versioning a Model
	4. Deploying a Model

## Introduction

Welcome to *Weakly Supervised Learning*. This is a free and open source book about building products using a part of the field of machine learning (ML) called natural language processing (NLP), deep learning (DL) and weakly supervised learning (WSL). WSL enables machines to learn without collecting and labeling millions of training records by hand. This book aims to provide a how to guide for shipping deep learning models using WSL.

As deep networks became state of the art they enabled new applications with unprecedented performance driven by unstructured data to proliferate. Unstructured data tends to dwarf structured data in size: many pages of text instead of columnar data, hours of audio instead of text, images instead of descriptions, video instead of images. And these algorithms are hungry: up to millions of unstructured records are required to achieve good performance on many tasks. While the open availability of large datasets and models based on them has helped feed networks, much of the work of building machine learning applications has become centered around data collection. Data alone however, is not enough as most tasks require labeled datasets. Work on machine learning products now involves expensive and time consuming curation of the labeled datasets that drive the models that drive products. The possession of strategic datasets is often a prerequisite for product innovation. 

This can be frustrating for a data scientist, machine learning engineer, product manager or aspiring ML entrepreneur because before they can use machine learning to build a product they have to find a way to gather or build something that generates lots of data! This means lots of jobs for machine learning engineers at growing companies with growing datasets, but it makes things hard for totally new applications. And new applications are what interest me.

Lorenzo Torresani in [Weakly Supervised Learning (2016)](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-31439-6_308) defines weakly supervised learning as:

> *Weakly supervised learning is a machine learning framework where the model is trained using examples that are only partially annotated or labeled.*  

What does it mean to me? Weakly supervised learning provides hope for the data poor. 

The book will explore practical applications of several methods of *weakly supervised learning* that have emerged in response to these developments. In *semi-supervised learning* an initial model trained on limited labeled data is used to label additional data, which then trains an improved final model. In *transfer learning* an existing model from a related domain is re-trained on or applied to training data from the problem domain. In *distant supervision* existing knowledge from databases and other sources is used to programmatically create low quality labels, which are combined via *weak supervision* in the form of a generative model into high quality labels for the entire dataset. I will demonstrate each strategy in the context of a deployable application.


![](DRAFT/yZmIEopL3sUmWDj7o4Z4aQA1qpDC9vYtZ2HaBk2MEiJQn3fpiCt2DvnBDsifgNeSRJuvfwdGgEXg_fASIIv6_sWt120BQLefSMAPwgxjOBf-bjgf57qsMZ3p4dKqSPQt1pVgBOZ4N_AQK7zsvQ.jpg)
![Weak Supervision: The New Programming Paradigm for Machine Learning (Ratner, Bach, Varma, Ré, et al)](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)
*Ask Hazy Research for permission*

In addition to demonstrating these methods, the book will also cover model versioning and management. The reader will learn to start with a relatively small labeled dataset and create a production grade model from concept through deployment.

## Statistical Natural Language Processing

*An empiricist approach to NLP suggests that we can learn the complicated and extensive structure by specifying an appropriate general language model, and then inducing the values of parameters by applying statistical, pattern recognition and machine learning methods to a large amount of language use.*

*—Foundations of Natural Language Processing*

Statistical Natural Language Processing (NLP)  is a field in which the structure of language is analyzed using statistical methods to learn a language model describing the detailed structure of natural language [Manning, Schütze, 1999]. Statistical language models are used to automate the processing of text in tasks including parsing sentences to extract their grammatical structure, extracting entities from documents. classifying documents into categories, ranking documents numerically, summarizing documents, answering questions, translating documents and others. 

Before the application of neural networks to language modeling, “core NLP techniques were dominated by machine-learning approaches that used linear models such as support vector machines or logistic regression, trained over very high dimensional yet very sparse feature vectors.” [Goldberg, 2015]  The primary challenge in the performance of these models was the *curse of dimensionality*, under which bag-of-words representations transformed to matrices via one-hot-encoding suffered from statistical insignificance across the deeply nested space in which they were encoded. Points become equidistant as more and more dimensions are added, so records appear more similar and there is not enough signal for an algorithm to work well. [Theodoridis, Koutroumbas, 2008] 

### Text Embeddings

In 2003, the paper Neural Probabilistic Language Model [Bengio, Ducharme, Vincent, Jauvin, 2003] demonstrated superior performance on several common NLP tasks using a distributed representation for words. A decade later, the rise of dense representations in the form of text embeddings like Word2Vec [Le, Mikolov, 2013] accelerated the development of DL methods for NLP. Text embeddings changed text encoding from a list of bits identifying the presence of words in a document under the bag-of-words model to a dense representation that describes the semantics of each word in terms of its position in a vector space where each dimension corresponds to a particular meaning [Tardy, 2017](https://www.quora.com/How-do-distributed-representation-avoid-the-curse-of-dimensionality-in-natural-language-processing-NLP/answer/Paul-Tardy). Neural networks work better with dense than with sparse representations. The chart below shows the difference between sparse and dense text feature representations.

![](DRAFT/86E6CAF6-064F-47C6-B355-F8F7CC91A6B1.png)
![Sparse vs. dense text encoding (Goldberg, 2015)](file://./images/intro/Sparse_vs_Dense_Embedding.png) 
*Ask Goldberg for permission*

### Convolutional Neural Networks

Convolutional Neural Networks are used for NLP tasks where local features are sufficient, such as document classification. As we’ll see, the most important signal for classifying documents are marker words and phrases common to documents of a given type and infrequent in the corpus overall. 

### Recurrent Neural Networks

Recurrent Neural Networks are used for NLP tasks where features need to be aware of a broader context within a sequence of words which is stored as internal state in each neuron and referenced in each output decision. 

## Snorkel
[Snorkel](https://www.snorkel.org/) is a [software project](https://github.com/snorkel-team/snorkel) originally from the Hazy Research group at Stanford University enabling the practice of *weak supervision*.  The project has an excellent [Get Started](https://www.snorkel.org/get-started/) page, and I recommend you spend some time working the [tutorials](https://github.com/snorkel-team/snorkel-tutorials) before proceeding beyond this chapter. 

Snorkel implements a generative model that accepts a matrix of weak labels for records in your training data and produces strong labels by learning the relationships between these weak labels.

### Labeling Functions (LFs)

A labeling function is a deterministic function used to label data as belonging to one class or another. They produce weak labels that in combination, through Snorkel’s generative models, can be used to generate strong labels for unlabeled data.

### Preprocessors

A preprocessor is a reusable function that maps a data point to another data point. It can be applied to the data before labeling functions so they can make use of external models or enable new labeling functions to work. For example, an address could be transformed into GPS coordinates, clustered, and a labeling function could be created based on the distribution of labels in terms of cluster membership: clusters with significantly more of one label could be labeled with a class, otherwise the LF could abstain.

### Data Augmentation with Transformation Functions (TFs)

Data augmentation is the use of functions that preprocess and transform the data so as to diversify the each data point and create a robust model. Transformation functions implement data augmentation in Snorkel. TFs take existing records and transform them into new records to enhance the label model.

### Slicing Functions (SFs)

Slicing functions enable us to focus on particular subsets of data that are more important to real world performance than others. For example part of a corpus of documents may be in a domain we’re familiar with, so we can gauge performance better by monitoring that slice of the data. Or our application and its data might group naturally into a few broad categories and we’re interested in monitoring them all independently.

## Posing a Problem
In this chapter we’re going to build a simple deep learning model to predict stack overflow tags from a sample of all questions ever asked and answered that have at least one point from upvotes. This isn’t obviously a small data problem where weakly supervised learning would be of interest, since the original dataset is multiple gigabytes and millions of records and a classifier should have no problem performing well using this data. As we’ll see, this is the case for frequent tags but when we try to extend to less frequent tags, we run into a sparsity and imbalance of data that makes weak supervision attractive.

## Exercising our GPU with some Natural Language Processing
Now that we’ve booted the cluster and service, let’s exercise it by training a neural network to tag StackOverflow questions. We treat this as a multi-class, multi-label problem. The training data has been balanced by upsampling the complete dump of questions that have at least one answer, one vote and have at least one tag occurring more than 2,000 times. It is about 600MB. This dataset was [previously computed](https://github.com/rjurney/deep_products/blob/master/code/stackoverflow/get_questions.spark.py) and the files can be found in the *paas_blog/data* directory of the Github repo.

You can view the Jupyter Notebook with the code we’ll be running from Github at [github.com/rjurney/paas_blog/DCOS_Data_Science_Engine.ipynb](https://github.com/rjurney/paas_blog/blob/master/DCOS_Data_Science_Engine.ipynb). We’ll be opening it using the JupyterLab Github interface, but if you like you can paste its content block-by-block into a new Python 3 notebook.

### Loading the Tutorial Notebook



### Verifying GPU Support

The first thing to do is to verify that our JupyterLab Python environment on our Data Science Engine EC2 instance is properly configured to work with its onboard GPU. We use `tensorflow.test.is_gpu_available` and `tensorflow.compat.v2.config.experimental.list_physical_devices` to verify the GPUs are working with Tensorflow.

```python
gpu_avail = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
print(f'1 or more GPUs is available: {gpu_avail}')

from tensorflow.python.client import device_lib
local_devices = device_lib.list_local_devices()
gpu = local_devices[3]
print(f"{gpu.name} is a {gpu.device_type} with {gpu.memory_limit / 1024 / 1024 / 1024:.2f}GB RAM")
```

You should see something like:

```python
1 or more GPUs is available: True
/device:GPU:0 is a GPU with 10.22GB RAM
```

### Loading the Data from S3

You can load the data for this tutorial using *pandas.read_parquet*.

```python
# Load the Stack Overflow questions right from S3
s3_parquet_path = f's3://{BUCKET}/08-05-2019/Questions.Stratified.Final.2000.parquet'
s3_fs = s3fs.S3FileSystem()

# Use pyarrow.parquet.ParquetDataset and convert to pandas.DataFrame
posts_df = ParquetDataset(
    s3_parquet_path,
    filesystem=s3_fs,
).read().to_pandas()

posts_df.head(3)
```

Now we load the indexes to convert back and forth between label indexes and text tags. We’ll use these to view the actual resulting tags predicted at the end of the tutorial.

```python
# Get the tag indexes
s3_client = boto3.resource('s3')

def json_from_s3(bucket, key):
    """Given a bucket and key for a JSON object, return the parsed object"""
    obj = s3_client.Object(bucket, key)
    obj.get()['Body'].read().decode('utf-8')
    json_obj = json.loads(obj.get()['Body'].read().decode('utf-8'))
    return json_obj


tag_index = json_from_s3(BUCKET, '08-05-2019/tag_index.2000.json')
index_tag = json_from_s3(BUCKET, '08-05-2019/index_tag.2000.json')

list(tag_index.items())[0:5], list(index_tag.items())[0:5]
```

Then we verify the number of records loaded:

```python
print(
    '{:,} Stackoverflow questions with a tag having at least 2,000 occurrences'.format(
        len(posts_df.index)
    )
)
```

```
1,554,788 Stackoverflow questions with a tag having at least 2,000 occurrences
```

### Preparing the Data

We need to join the previously tokenized text back into a string for use in a Tokenizer, which provides useful properties. In addition, making the number of documents a multiple of batch size is a requirement for Tensorflow/Keras to split work among multiple GPUs and to use certain models such as Elmo.

```python
import math

BATCH_SIZE = 64
MAX_LEN = 200
TOKEN_COUNT = 10000
EMBED_SIZE = 50
TEST_SPLIT = 0.2

# Convert label columns to numpy array
labels = posts_df[list(posts_df.columns)[1:]].to_numpy()

# Training_count must be a multiple of the BATCH_SIZE times the MAX_LEN
highest_factor = math.floor(len(posts_df.index) / (BATCH_SIZE * MAX_LEN))
training_count = highest_factor * BATCH_SIZE * MAX_LEN
print(f'Highest Factor: {highest_factor:,} Training Count: {training_count:,}')

# Join the previously tokenized data for tf.keras.preprocessing.text.Tokenizer to work with
documents = []
for body in posts_df[0:training_count]['_Body'].values.tolist():
    words = body.tolist()
    documents.append(' '.join(words))

labels = labels[0:training_count]

# Conserve RAM
del posts_df
gc.collect()

# Lengths for x and y match
assert( len(documents) == training_count == labels.shape[0] )
```

You should see:

```
Highest Factor: 121 Training Count: 1,548,800
```

### Pad the Sequences

The data has already been truncated to 200 words per post but the tokenization using the top 10K words reduces this to below 200 in some documents. If any documents vary from 200 words, the data won't convert properly into a *numpy* matrix below. 

In addition to converting the text to numeric sequences with a key, Keras’ *Tokenizer* class is handy for producing the final results of the model via the [*keras.preprocessing.text.Tokenizer.sequences_to_texts*](https://keras.io/preprocessing/text/#tokenizer) method. Then we use Keras’ [*keras.preprocessing.sequence.pad_sequences*](https://keras.io/preprocessing/sequence/#pad_sequences) method and check the output to ensure the sequences are all 200 items long or they won’t convert properly into a matrix. The string `__PAD__` has been used previously to pad the documents, so we reuse it here.

```python
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(
    num_words=TOKEN_COUNT,
    oov_token='__PAD__'
)
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    dtype='int32',
    padding='post',
    truncating='post',
    value=1
)

# Conserve RAM
del documents
del sequences
gc.collect()

print( max([len(x) for x in padded_sequences]), min([len(x) for x in padded_sequences]) )
assert( min([len(x) for x in padded_sequences]) == MAX_LEN == max([len(x) for x in padded_sequences]) )

padded_sequences.shape
```

### Split into Test/Train Datasets

We need one dataset to train with and one separate dataset to test and validate our model with.  The oft used [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) makes it so.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    labels,
    test_size=TEST_SPLIT,
    random_state=1337
)

# Conserve RAM
del padded_sequences
del labels
gc.collect()

assert(X_train.shape[0] == y_train.shape[0])
assert(X_train.shape[1] == MAX_LEN)
assert(X_test.shape[0] == y_test.shape[0]) 
assert(X_test.shape[1] == MAX_LEN)
```

### Compute Class Weights

Although there has already been filtering and up-sampling of the data to restrict it to a sample of questions with at least one tag that occurs more than 2,000 times, there are still uneven ratios between common and uncommon labels. Without class weights, the most common label will be much more likely to be predicted than the least common. Class weights will make the loss function consider uncommon classes more than frequent ones.

```python
train_weight_vec = list(np.max(np.sum(y_train, axis=0))/np.sum(y_train, axis=0))
train_class_weights = {i: train_weight_vec[i] for i in range(y_train.shape[1])}

sorted(list(train_class_weights.items()), key=lambda x: x[1])[0:10]
```

### Train a Classifier Model to Tag Stack Overflow Posts

Now we’re ready to train a model to classify/label questions with tag categories. The model is based on [Kim-CNN](https://arxiv.org/abs/1408.5882), a commonly used convolutional neural network for sentence and document classification. We use the functional API and we’ve heavily parametrized the code so as to facilitate experimentation. 

```python
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Dense, Activation, Embedding, Flatten, MaxPool1D, GlobalMaxPool1D, Dropout, Conv1D, Input, concatenate
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

FILTER_LENGTH       = 300
FILTER_COUNT        = 128
FILTER_SIZES        = [3, 4, 5]
EPOCHS              = 4
ACTIVATION          = 'selu'
CONV_PADDING        = 'same'
EMBED_SIZE          = 50
EMBED_DROPOUT_RATIO = 0.1
CONV_DROPOUT_RATIO  = 0.1
LOSS                = 'binary_crossentropy'
OPTIMIZER           = 'adam'
```

In Kim-CNN, we start by encoding the sequences using an *Embedding*, followed by a *Dropout* layer to reduce overfitting. Next we split the graph into multiple *Conv1D* layers with different widths, each followed by *MaxPool1D*. These are joined by concatenation and are intended to characterize patterns of different size sequence lengths in the documents. There follows another *Conv1D*/*GlobalMaxPool1D* layer to summarize the most important of these patterns. This is followed by flattening into a *Dense* layer and then on to the final *sigmoid* output layer. Otherwise we use *selu* throughout.

```python
padded_input = Input(
    shape=(X_train.shape[1],), 
    dtype='int32'
)

# Create an embedding with RandomUniform initialization
emb = Embedding(
    TOKEN_COUNT, 
    EMBED_SIZE, 
    input_length=X_train.shape[1],
    embeddings_initializer=RandomUniform()
)(padded_input)
drp = Dropout(EMBED_DROPOUT_RATIO)(emb)

# Create convlutions of different kernel sizes
convs = []
for filter_size in FILTER_SIZES:
    f_conv = Conv1D(
        filters=FILTER_COUNT,
        kernel_size=filter_size,
        padding=CONV_PADDING,
        activation=ACTIVATION
    )(drp)
    f_pool = MaxPool1D()(f_conv)
    convs.append(f_pool)

l_merge = concatenate(convs, axis=1)
l_conv = Conv1D(
    128,
    5,
    activation=ACTIVATION
)(l_merge)
l_pool = GlobalMaxPool1D()(l_conv)
l_flat = Flatten()(l_pool)
l_drop = Dropout(CONV_DROPOUT_RATIO)(l_flat)
l_dense = Dense(
    128,
    activation=ACTIVATION
)(l_drop)
out_dense = Dense(
    y_train.shape[1],
    activation='sigmoid'
)(l_dense)

model = Model(inputs=padded_input, outputs=out_dense)
```

Next we compile our model. We use a variety of metrics, because no one metric summarizes model performance, and we need to drill down into the true and false positives and negatives. We also use the *ReduceLROnPlateau*, *EarlyStopping* and *ModelCheckpoint* callbacks to improve performance once we hit a plateau, then to stop early, and to persist only the very best model in terms of the validation categorical accuracy. 

Categorical accuracy is the best fit for gauging our model’s performance because it gives points for each row separately for each class we’re classifying. This means that if we miss one, but get the others right, this is a great result. With binary accuracy, the entire row is scored as incorrect.

Then it is time to fit the model. We give it the class weights we computed earlier.

```python
model.compile(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalseNegatives(),
    ]
)
model.summary()

callbacks = [
    ReduceLROnPlateau(
        monitor='val_categorical_accuracy',
        factor=0.1,
        patience=1,
    ), 
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=2
    ), 
    ModelCheckpoint(
        filepath='kim_cnn_tagger.weights.hdf5',
        monitor='val_categorical_accuracy',
        save_best_only=True
    ),
]

history = model.fit(X_train, y_train,
                    class_weight=train_class_weights,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks)
```

### Load the Best Model from Training Epochs

Because we used `ModelCheckpoint(save_only_best=True)`, the best epoch in terms of `CategoricalAccuracy` is what was saved. We want to use that instead of the last epoch's model, which is what is stored in `model` above. So we load the file before evaluating our model.

```python
model = tf.keras.models.load_model('kim_cnn_tagger.weights.hdf5')
metrics = model.evaluate(X_test, y_test)
```

### Parse and Print Final Metrics

Metrics include names like *precision_66* which aren't consistent between runs. We fix these to cleanup our report on training the model. We also add an f1 score, then make a DataFrame to display the log. This could be extended in repeat experiments.

```python
def fix_metric_name(name):
    """Remove the trailing _NN, ex. precision_86"""
    if name[-1].isdigit():
        repeat_name = '_'.join(name.split('_')[:-1])
    else:
        repeat_name = name
    return repeat_name

def fix_value(val):
    """Convert from numpy to float"""
    return val.item() if isinstance(val, np.float32) else val

def fix_metric(name, val):
    repeat_name = fix_metric_name(name)
    py_val = fix_value(val)
    return repeat_name, py_val

log = {}
for name, val in zip(model.metrics_names, metrics):
    repeat_name, py_val = fix_metric(name, val)
    log[repeat_name] = py_val
log.update({'f1': (log['precision'] * log['recall']) / (log['precision'] + log['recall'])})

pd.DataFrame([log])
```

### Plot the Epoch Accuracy

We want to know the performance at each epoch so that we don't train needlessly large numbers of epochs. 

```python
%matplotlib inline

new_history = {}
for key, metrics in history.history.items():
    new_history[fix_metric_name(key)] = metrics

import matplotlib.pyplot as plt

viz_keys = ['val_categorical_accuracy', 'val_precision', 'val_recall']
# summarize history for accuracy
for key in viz_keys:
    plt.plot(new_history[key])
plt.title('model accuracy')
plt.ylabel('metric')
plt.xlabel('epoch')
plt.legend(viz_keys, loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

### Check the Actual Prediction Outputs

It is not enough to know theoretical performance. We need to see the actual output of the tagger at different confidence thresholds.

```python
TEST_COUNT = 1000

X_test_text = tokenizer.sequences_to_texts(X_test[:TEST_COUNT])

y_test_tags = []
for row in y_test[:TEST_COUNT].tolist():
    tags = [index_tag[str(i)] for i, col in enumerate(row) if col == 1]
    y_test_tags.append(tags)

CLASSIFY_THRESHOLD = 0.5

y_pred = model.predict(X_test[:TEST_COUNT])
y_pred = (y_pred > CLASSIFY_THRESHOLD) * 1

y_pred_tags = []
for row in y_pred.tolist():
    tags = [index_tag[str(i)] for i, col in enumerate(row) if col > CLASSIFY_THRESHOLD]
    y_pred_tags.append(tags)
```

Lets look at the sentences with the actual labels and the predicted labels in a *DataFrame*:

```python
prediction_tests = []
for x, y, z in zip(X_test_text, y_pred_tags, y_test_tags):
    prediction_tests.append({
        'Question': x,
        'Predictions': ' '.join(sorted(y)),
        'Actual Tags': ' '.join(sorted(z))
    })

pd.DataFrame(prediction_tests)
```

We can see from these three records that the model is doing fairly well. This tells a different story than performance metrics alone. It is so strange that most machine learning examples just compute performance and don’t actually employ the `predict()` method! At the end of the day statistical performance is irrelevant and what matters is the real world performance - which is not contained in simple summary statistics!

![](DRAFT/jupyter_results.png)


# Bibliography
* Torresani Lorenzo,  [Computer Vision](https://link.springer.com/referencework/10.1007/978-0-387-31439-6), [“Weakly Supervised Learning”](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-31439-6_308), Springer, Boston, MA, USA, 2016
* Ratner, Bach, Varma, Ré, et al, [“Weak Supervision: The New Programming Paradigm for Machine Learning”](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html), *Hazy Research*
* Manning, Christopher D., Schiitze, Hinrich, [Foundations of Statistical Natural Language Processing](https://amzn.to/2HRDFBm), The MIT Press, Cambridge, MA, USA, 1999
* Goldberg, Yoav, [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), 2015
* Theodoridis, Koutroumbas, [Pattern Recognition - 4th Edition](https://www.elsevier.com/books/pattern-recognition/theodoridis/978-1-59749-272-0), Academic Press, Millbrae, CA, USA, 2008
* Bengio, Ducharme, Vincent, Jauvin, [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), Journal of Machine Learning Research, 2003
* Le, Mikolov, [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), Google Inc, Mountain View, CA, USA, 2013
* 

## Semi-Supervised Learning Quotes

(Quotes from [Semi-Supervised Learning (2005)](http://www.acad.bg/ebook/ml/MITPress-%20SemiSupervised%20Learning.pdf)):

“there is an important prerequisite: that the distribution of examples, which the unlabeled data will help elucidate, be relevant for the classification problem. In a more mathematical formulation, one could say that the knowledge on p(x) that one gains through the unlabeled data has to carry information that is useful in the inference of p(y|x)."
"Smoothness assumption of supervised learning: If two points x1, x2 are close, then so should be the corresponding outputs y1, y2."
"The assumption is that the label function is smoother in high-density smoothness assumption regions than in low-density regions: Semi-supervised smoothness assumption: If two points x1, x2 in a high-density region are close, then so should be the corresponding outputs y1, y2."
"Note that by transitivity, this assumption implies that if two points are linked by a path of high density (e.g., if they belong to the same cluster), then their outputs are likely to be close. If, on the other hand, they are separated by a low-density region, then their outputs need not be close."
"Cluster assumption: If points are in the same cluster, they are likely to be of the same class."
"Low density separation: The decision boundary should lie in a low-density region."

![How to get more training data?](images/intro/WS_mapping.png)

## Approach

As in my books Agile Data Science and Agile Data Science 2.0, we’ll be doing things iteratively with an end to end product in order to get an MVP we can show users sooner.

... you can't monitor patents or you could end up knowingly infringing. But could you monitor a summary/insights?

## Noteworthy Links

* [Understanding LSTM Networks (2015)][5] - good internal figures of RNN/LSTM
* [The Unreasonable Effectiveness of Recurrent Neural Networks (2015)][6] - an excellent introduction to RNNs and LSTMs with excellent figures. This gives you intuitive examples about what RNN/LSTM are able to learn working from the ground up.
	"If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs."
	"...even if your data is not in form of sequences, you can still formulate and train powerful models that learn to process it sequentially. You’re learning stateful programs that process your fixed-sized data."
* [DeepDive: A Distant Supervision Framework](http://deepdive.stanford.edu/)
* [NLP’s ImageNet Moment has Arrived](http://ruder.io/nlp-imagenet/)
* [Building NLP Classifiers Cheaply With Transfer Learning and Weak Supervision](https://towardsdatascience.com/a-technique-for-building-nlp-classifiers-efficiently-with-transfer-learning-and-weak-supervision-a8e2f21ca9c8) 
* [Data Augmentation in NLP - Towards Data Science](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)
* [GitHub - jasonwei20/eda_nlp: Code for the EMNLP-IJCNLP paper: Easy data augmentation techniques for boosting performance on text classification tasks.](https://github.com/jasonwei20/eda_nlp)
* 

## Themes

* Using high level libraries in chains to rapidly build baseline models: fasttext, auto\_ml, etc. Find and string together.
* Full development lifecycle considerations - from model development to model deployment - do I really want to address this?
* Minimizing the amount of labeled training data required via unsupervised learning, semi-supervised training, [Snorkel Drybell][7], etc.

## Libraries

* [MLFlow][8] - this can be used to deploy and monitor models!
* [Apache Superset][9]
* [Google Knowledge Graph Search API][10]
* [FastText][11]

### Parsing Python Code

* [parso - A Python Parser — parso 0.5.0 documentation](https://parso.readthedocs.io/en/latest/)

### Graphs

* [Stellargraph][12]

### SPARQL on Wikidata

* [Where do Mayors Come From: Querying Wikidata with Python and SPARQL - Parametric Thoughts](https://janakiev.com/blog/wikidata-mayors/)
* 

## Semi-Supervised Learning
Use Glove, etc. embeddings and then customize for your domain. Does this work for code embeddings?

* [Snorkel Drybell][13]

### Word/Code Embeddings

* [GloVe][14] - source code for C version. Global Vectors for Word Representation.

* [anaGo][15] - a Python library for sequence labeling(NER, PoS Tagging, semantic role labeling (SRL)), implemented in Keras.

* [source2vec](https://nbviewer.jupyter.org/github/Jur1cek/source2vec/blob/master/load_embeddings.ipynb) - Python and other language embeddings using word2vec, glove, fastext, etc. See also [GitHub - Jur1cek/source2vec: Source code embeddings for various programming languages](https://github.com/Jur1cek/source2vec)

* [Phrase Detection Using Gensim Parser | Shane Lynn](https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/)

* [GitHub - artetxem/vecmap: A framework to learn cross-lingual word embedding mappings](https://github.com/artetxem/vecmap) - potential Python to English word embedding mapping



## NLP/Weak Supervision Applications of StackExchange Data
### Code Embeddings

* Cluster code examples? Relationship with tags?
* Recommend code snippets
* Labeler for StackExchange Questions using tag—>entity->property relationships from Wikidata/Sparql -  “high level programming language”, create labels using knowledge base! But to what end? What is the goal?

### Text Classification and Categorization

* information filtering
* sentiment analysis
* paraphrase detection
* knowledge extraction - RDF like facts about languages/posts based on tag/entity mappings?


* Language Modeling
* Caption Generation
* Document Summarization

### Word Level Classification

* Named Entity Resolution

### Sentence Level Classification

* Sentiment Analysis

* Semantic matching between texts

### Generating Language - Encoders - Decoders

* Machine Translation - seq2seq - translate one language stackoverflow to another
* Image Captioning
* Question Answering

### Deep Clustering

* [How to do Unsupervised Clustering with Keras][16]
* Neural Networks with Keras Cookbook, Chapter 11

### Transfer Learning
* Using BERT transfer then training embedding on our data - [A Comprehensive Hands on Guide to Transfer Learning with Real World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a)
* 
