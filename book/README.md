# Weakly Supervised Learning: Do More with Less Data
#  Table of Contents

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

# Introduction

Welcome to *Weakly Supervised Learning*. This is a free and open source book about building products using a part of the field of machine learning (ML) called natural language processing (NLP), deep learning (DL) and weakly supervised learning (WSL). WSL enables machines to learn without collecting and labeling millions of training records by hand. This book aims to provide a how to guide for shipping deep learning models using WSL.



# Chapter 1: Weakly Supervised Learning
Welcome to the chapter length introduction to weakly supervised learning! This is the section where I give you the context that will inspire and motivate you to read the rest of this book: the overview. Right? This is a strange exercise for me, because it involves citing a great number of academic publications, something I’ve never done before. I have tried to keep things simple enough that a business audience might get value while still covering the material in sufficient depth. I hope this high level explanation of how we (and I mean we as in I follow along while researchers invent) got to where we are in deep learning and natural language processing serves as an introduction for some and as a worthwhile review for others. I’m a history nerd, so let’s start with some recent history.

Since deep learning became state of the art in several areas in 2011/12, deep networks have enabled new applications with unprecedented performance driven by unstructured data to proliferate in research and industry. Unstructured data tends to dwarf structured data in size: many pages of text instead of columnar data, hours of audio instead of text, images instead of descriptions, video instead of images. And these algorithms are hungry: up to millions of unstructured records are required to achieve good performance on many tasks, and this is increases with the complexity of the task. While the open availability of large datasets and models based on them has helped feed networks, much of the work of building machine learning applications is now centered around data collection. Data alone however, is not enough.

While unsupervised learning is increasingly powerful and the most sophisticated language models now work this way, most end applications involve supervised learning. Supervised learning requires labeled datasets, and labeling data is very expensive or even impossible. Dave McCrory, who as VP of Software Engineering at GE Digital developed AI applications for an array of industries, recently told me, “For medical diagnosis, hand labeling large numbers of MRIs by a doctor isn’t just expensive, it’s impossible. A radiologist will not label more than about a thousand records before he refuses to do more.”  Work on machine learning products now involves expensive and time consuming curation of the labeled datasets that drive the models that drive products. The possession of strategic datasets is often a prerequisite for product innovation. 

Curation isn’t limited to just data - companies now manage large workforces for hand labeling data. “Mechanical turks” or task workers have to be rigorously selected and trained as well. Some companies and products - such as Amazon’s Mechanical Turk - create elaborate systems of tests to qualify task workers for a set of labeling tasks. A user might be required to have a certain minimum accuracy score on a set of standard tests or to achieve a certain level of performance across multiple metrics on real world tasks. Passing these tests represents a serious investment of time for both the worker and the company, and the process is incredibly competitive globally.

At any given moment there are over 100,000 workers using the Mechanical Turk platform [Djellel, Difallah; Filatova, Elena; Ipeirotis, Panos, 2018], and each worker’s labels are compared to known good results (also known as “gold standard data”) or several other workers’ results on the same task to determine their performance. This system can be brutal because abstracted behind a programmatic API, task workers are intelligent human beings performing tasks that are mindless for the end user (is this a cat? Is this a stoplight?) who is primarily tasked with labeling data because he or she lacks economic opportunity compared to the hiring party. Some companies recruit and pay task workers hourly through websites such as UpWork, others hire them directly in an office environment. Tasks issued to workers hired by both methods of recruitment are becoming increasingly sophisticated, so as some point of sophistication it makes sense to form an ongoing relationship with task workers as employees of an organization to be trained, developed and retained. In short, armies of task workers and the hardware and software systems, processes and management supporting them represent a major cost in the development of artificial intelligence applications. This cost can be a major predictor of AI project success. Chris Albion, who leads the data science team at Devoted Health, said in 2019, “The sheer volume of data you need means that it better be cheap as dirt to collect. If you hear, ‘Our model is trained using data from vaporizing moon rocks’, that is going to be a terrible startup.”

Sophisticated software platforms such as reCAPTCHA and startups selling them have arisen to try to reduce labeling costs, an indication of the scale of the problem. Often these systems are built into features of products. The cost of acquiring labeled training data is otherwise so high that so far it is the major consumer internet players who have dominated the space because data collection is built into their products. This has also led to the co-development of the sophisticated infrastructure driving big data, which consumer internet companies had first (after science) and which they have productized as the leaders of the cloud computing market. Investor and veteran data scientist Pete Skomoroch said in 2018, “Without large amounts of good raw and labeled training data, solving most AI problems is not possible. Acquiring the data is harder to speed up. The best companies have data collection built into [their] product [and user experience] and get AI training data from their users.” He went on to say in 2019, “Data labeling is a good proxy for whether machine learning is cost effective for a problem. If you can build labeling into normal user activities you track like Facebook, Google and Amazon consumer applications you have a shot. Otherwise, you burn money paying for labeled data.”

This can be frustrating for a data scientist, machine learning engineer, product manager, chief data or aspiring ML entrepreneur because before they can use machine learning to build a product they have to find a way to gather or build something that generates lots of data! This means lots of jobs for machine learning engineers at growing companies with growing datasets, but it makes things hard for totally new applications. And new applications are what interest me.

Lorenzo Torresani in [Weakly Supervised Learning (2016)](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-31439-6_308) defines weakly supervised learning as:

> *Weakly supervised learning is a machine learning framework where the model is trained using examples that are only partially annotated or labeled.*  

What does it mean to me? Weakly supervised learning provides hope for the data poor. 

The book will explore practical applications of several methods of *weakly supervised learning* that have emerged in response to these developments. In *semi-supervised learning* an initial model trained on limited labeled data is used to label additional data, which then trains an improved final model. In *transfer learning* an existing model from a related domain is re-trained on or applied to training data from the problem domain. In *distant supervision* existing knowledge from databases and other sources is used to programmatically create low quality labels, which are combined via *weak supervision* in the form of a generative model into high quality labels for the entire dataset. I will demonstrate each strategy in the context of a deployable application.


![](README/yZmIEopL3sUmWDj7o4Z4aQA1qpDC9vYtZ2HaBk2MEiJQn3fpiCt2DvnBDsifgNeSRJuvfwdGgEXg_fASIIv6_sWt120BQLefSMAPwgxjOBf-bjgf57qsMZ3p4dKqSPQt1pVgBOZ4N_AQK7zsvQ%204.jpg)
![Weak Supervision: The New Programming Paradigm for Machine Learning (Ratner, Bach, Varma, Ré, et al)](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)

<REMINDER: Ask Hazy Research for permission>

In addition to demonstrating these methods, the book will also cover model versioning and management. The reader will learn to start with a relatively small labeled dataset and create a production grade model from concept through deployment.

# Statistical Natural Language Processing

*An empiricist approach to NLP suggests that we can learn the complicated and extensive structure by specifying an appropriate general language model, and then inducing the values of parameters by applying statistical, pattern recognition and machine learning methods to a large amount of language use.*

*—Foundations of Natural Language Processing*

Statistical Natural Language Processing (NLP)  is a field in which the structure of language is analyzed using statistical methods to learn a language model describing the detailed structure of natural language [Manning, Schütze, 1999]. Statistical language models are used to automate the processing of text in tasks including parsing sentences to extract their grammatical structure, extracting entities from documents. classifying documents into categories, ranking documents numerically, summarizing documents, answering questions, translating documents and others. 

Before the application of neural networks to language modeling, “core NLP techniques were dominated by machine-learning approaches that used linear models such as support vector machines or logistic regression, trained over very high dimensional yet very sparse feature vectors.” [Goldberg, 2015]  The primary challenge in the performance of these models was the *curse of dimensionality*, under which bag-of-words representations transformed to matrices via one-hot-encoding suffered from statistical insignificance across the deeply nested space in which they were encoded. Points become equidistant as more and more dimensions are added, so records appear more similar and there is not enough signal for an algorithm to work well [Theodoridis, Koutroumbas, 2008].

## Text Embeddings

In 2003, the paper Neural Probabilistic Language Model [Bengio, Ducharme, Vincent, Jauvin, 2003] demonstrated superior performance on several common NLP tasks using a distributed representation for words. A decade later, the rise of dense representations in the form of text embeddings like Word2Vec [Le, Mikolov, 2013] accelerated the development of DL methods for NLP. Text embeddings changed text encoding from a list of bits identifying the presence of words in a document under the bag-of-words model to a dense representation that describes the semantics of each word in terms of its position in a vector space where each dimension corresponds to a particular meaning [Tardy, 2017](https://www.quora.com/How-do-distributed-representation-avoid-the-curse-of-dimensionality-in-natural-language-processing-NLP/answer/Paul-Tardy). Neural networks work better with dense than with sparse representations. The chart below shows the difference between sparse and dense text feature representations.

![](README/86E6CAF6-064F-47C6-B355-F8F7CC91A6B1%204.png)
![Sparse vs. dense text encoding (Goldberg, 2015)](file://./images/intro/Sparse_vs_Dense_Embedding.png) 

<REMINDER: Ask Goldberg for permission>

## Convolutional Neural Networks

Convolutional Neural Networks are used for NLP tasks where local features are sufficient, such as document classification. As we’ll see, the most important signal for classifying documents are marker words and phrases common to documents of a given type and infrequent in the corpus overall. 

## Recurrent Neural Networks

Recurrent Neural Networks are used for NLP tasks where features need to be aware of a broader context within a sequence of words which is stored as internal state in each neuron and referenced in each output decision. 

# Snorkel
[Snorkel](https://www.snorkel.org/) is a [software project](https://github.com/snorkel-team/snorkel) originally from the Hazy Research group at Stanford University enabling the practice of *weak supervision*.  The project has an excellent [Get Started](https://www.snorkel.org/get-started/) page, and I recommend you spend some time working the [tutorials](https://github.com/snorkel-team/snorkel-tutorials) before proceeding beyond this chapter. 

Snorkel implements a generative model that accepts a matrix of weak labels for records in your training data and produces strong labels by learning the relationships between these weak labels.

## Labeling Functions (LFs)

A labeling function is a deterministic function used to label data as belonging to one class or another. They produce weak labels that in combination, through Snorkel’s generative models, can be used to generate strong labels for unlabeled data.

## Preprocessors

A preprocessor is a reusable function that maps a data point to another data point. It can be applied to the data before labeling functions so they can make use of external models or enable new labeling functions to work. For example, an address could be transformed into GPS coordinates, clustered, and a labeling function could be created based on the distribution of labels in terms of cluster membership: clusters with significantly more of one label could be labeled with a class, otherwise the LF could abstain.

## Data Augmentation with Transformation Functions (TFs)

Data augmentation is the use of functions that preprocess and transform the data so as to diversify the each data point and create a robust model. Transformation functions implement data augmentation in Snorkel. TFs take existing records and transform them into new records to enhance the label model.

## Slicing Functions (SFs)

Slicing functions enable us to focus on particular subsets of data that are more important to real world performance than others. For example part of a corpus of documents may be in a domain we’re familiar with, so we can gauge performance better by monitoring that slice of the data. Or our application and its data might group naturally into a few broad categories and we’re interested in monitoring them all independently.

## Chapter Bibliography

* Torresani Lorenzo,  [Computer Vision](https://link.springer.com/referencework/10.1007/978-0-387-31439-6), [“Weakly Supervised Learning”](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-31439-6_308), Springer, Boston, MA, USA, 2016
* Ratner, Bach, Varma, Ré, et al, [“Weak Supervision: The New Programming Paradigm for Machine Learning”](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html), *Hazy Research*
* Manning, Christopher D., Schiitze, Hinrich, [Foundations of Statistical Natural Language Processing](https://amzn.to/2HRDFBm), The MIT Press, Cambridge, MA, USA, 1999
* Goldberg, Yoav, [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726), 2015
* Theodoridis, Koutroumbas, [Pattern Recognition - 4th Edition](https://www.elsevier.com/books/pattern-recognition/theodoridis/978-1-59749-272-0), Academic Press, Millbrae, CA, USA, 2008
* Bengio, Ducharme, Vincent, Jauvin, [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), Journal of Machine Learning Research, 2003
* Le, Mikolov, [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), Google Inc, Mountain View, CA, USA, 2013
* Konstantinos, Sechidis, Tsoumakas, Grigorios, Vlahavas, Ioannis, [On the Stratification of Multi-Label Data](On the Stratification of Multi-Label Data), Aristotle University of Thessaloniki, Thessaloniki, Geece, 2011
* Wikipedia contributors, “Stack Exchange.” *Wikipedia, The Free Encyclopedia*. Wikipedia, The Free Encyclopedia, 18 Sep. 2019. Accessed 25 Sep. 2019
* Djellel, Difallah; Filatova, Elena; Ipeirotis, Panos (2018). [Demographics and dynamics of mechanical turk workers](http://www.ipeirotis.com/wp-content/uploads/2017/12/wsdmf074-difallahA.pdf) *Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining*: 135–143.

# Chapter 2: Environment Setup
In this chapter we will recreate the environment in which the book’s examples were created so that you can run them without any problems. I’ve created Conda and Virtual Environments for you to use to run the Jupyter Notebooks that contain the book’s examples.

In my previous book I setup EC2 and Vagrant environments in which to run the book’s code but since 2017 the Python ecosystem has developed to the point that I am going to refrain from providing thorough installation documentation for every requirement. The website for each library is a better resource than I can possibly create, and they are updated and maintained more frequently than this book. I will instead list requirements, link to the project pages and let the reader install the requirements themselves.

## Requirements
* Linux, Mac OS X or Windows
* [Git](https://git-scm.com/download) is used to check out the book’s source code
* Python 3.6+ - I recommend [Anaconda Python](https://www.anaconda.com/distribution/), but any Python will do
* An NVIDIA graphics card - you can work the examples without one, but CPU training is painfully slow
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) - for GPU acceleration in CuPy and Tensorflow
* [cuDNN](https://developer.nvidia.com/cudnn) - for GPU acceleration in Tensorflow
* [LibYAML](https://pyyaml.org/wiki/LibYAML) - YAML Library

## Environment Setup
I’ve defined two Python environments for the book using Conda and a Virtual Environment. Once you have setup the requirements, you can easily reproduce the environment in which the book was written and tested.

While it is possible to run this code on any environment, Mac OS X has far more problems with Python support for machine learning libraries than does Linux (particularly graphics cards for GPU acceleration), so you if you have a Mac you might want to think about using a Linux machine to run the examples because there is no guarantee the environment won’t break at any time because the wind blows just right.

While the book does include support for Windows users, be aware that while I have extensive experience with operating systems resembling BSD and System V Unix such as (Mac OS X and Linux), I am not as experienced with Windows. This means that for Windows are more likely, and for that I apologize. I have done my best to thoroughly test and keep things working well.

### Getting the Code

You will need git to check out the code in the [Github repository for this book](https://github.com/rjurney/deep_products). If you don’t have it installed, see [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

```bash
git clone https://github.com/rjurney/weakly_supervised_learning
```

You can view all the examples and the complete book in the Jupyter Notebooks under *weakly_supervised_learning/code*.

### Setting Up a Conda Environment

To run the code, you’ll need to setup a Python environment with all the dependencies installed.

To create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) you will need Anaconda Python (see [Requirements](##Requirements)). Use *weakly_supervised_learning/code/environment.yml* to create the environment:

```bash
conda create -n weak -f environment.yml -y
```

To activate this environment, run:

```bash
conda activate weak
```

To exit this environment, run:

```bash
conda deactivate
```

### Setting Up a Virtual Environment

Alternatively you can use [virtualenv](https://virtualenv.pypa.io/en/latest/) and [pip](https://pypi.org/project/pip/). To setup the environment and install packages, run:

```bash
# Add '-p </path/to/my/python>' to specify a Python other than the one in PATH
virtualenv venv
```

To activate this environment, run:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

To exit this environment, run:

```bash
source deactivate
```

### Running Jupyter

![](README/example_jupyter_notebook%204.png)

[Jupyter](https://jupyter.org/) is installed as part of creating the Python environment. To run Jupyter, you can run:

```bash
cd </path/to/weakly_supervised_learning/code>
jupyter notebook &
```

Then visit [http://localhost:8888](http://localhost:8888) and select the chapter file you want to read and run.

## The Dataset
Stack Overflow was started in 2008 by as a website 

Stack Exchange publishes their entire database interactively via the [Stack Exchange Data Explorer](https://data.stackexchange.com/) or in bulk from the [internet archive](https://archive.org/download/stackexchange). This is not small data. The posts from the September 2, 2019 dump are 13.9GB compressed, and are stored as a single XML file with one tag for each post.

```
stackoverflow.com-Badges.7z 02-Sep-2019 13:10 233.9M
stackoverflow.com-Comments.7z 02-Sep-2019 13:22 4.1G
stackoverflow.com-PostHistory.7z 03-Sep-2019 23:59 24.4G
stackoverflow.com-PostLinks.7z 03-Sep-2019 16:11 82.1M
stackoverflow.com-Posts.7z 04-Sep-2019 14:38 13.9G
stackoverflow.com-Tags.7z 03-Sep-2019 16:11 787.6K
stackoverflow.com-Users.7z 03-Sep-2019 16:27 477.2M
stackoverflow.com-Votes.7z 03-Sep-2019 16:34 1.0G
```
*Files for English Stack Overflow dump for September 3rd, 2019*

```xml
<U+FEFF><?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="4" PostTypeId="1" AcceptedAnswerId="7" CreationDate="2008-07-31T21:42:52.667" Score="630" ViewCount="42817" Body="&lt;p&gt;I want to use a track-bar to change a form's opacity.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;This is my code:&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;decimal trans = trackBar1.Value / 5000;&#xA;this.Opacity = trans;&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;When I build the application, it gives the following error:&lt;/p&gt;&#xA;&#xA;&lt;blockquote&gt;&#xA;  &lt;p&gt;Cannot implicitly convert type &lt;code&gt;'decimal'&lt;/code&gt; to &lt;code&gt;'double'&lt;/code&gt;&lt;/p&gt;&#xA;&lt;/blockquote&gt;&#xA;&#xA;&lt;p&gt;I tried using &lt;code&gt;trans&lt;/code&gt; and &lt;code&gt;double&lt;/code&gt; but then the control doesn't work. This code worked fine in a past VB.NET project.&lt;/p&gt;&#xA;" OwnerUserId="8" LastEditorUserId="3641067" LastEditorDisplayName="Rich B" LastEditDate="2019-07-19T01:39:54.173" LastActivityDate="2019-07-19T01:39:54.173" Title="Convert Decimal to Double?" Tags="&lt;c#&gt;&lt;floating-point&gt;&lt;type-conversion&gt;&lt;double&gt;&lt;decimal&gt;" AnswerCount="13" CommentCount="2" FavoriteCount="48" CommunityOwnedDate="2012-10-31T16:42:47.213" />
...
</posts>
```
*Top/bottom of Stack Overflow dump file Posts.xml*

This isn’t an obvious small data problem where supervision would be of interest, since the original dataset is over ten gigabytes and millions of records. Our stratified sample of questions is half a gigabyte compressed and consists of 1.5 million questions that a classifier should have no problem tagging. As we’ll see, this is the case for frequent tags but when we try to extend coverage to less frequent tags, we run into sparsity and imbalanced data that create problems which make weak supervision attractive. We’ll be using weakly supervised learning to improve the model’s breadth rather than just its categorical accuracy on all tags. This process will use transfer learning, semi-supervised learning, weak supervision, distant supervision, and the [Snorkel](https://www.snorkel.org/) software package.

## Multi-label? Sort of.
There’s another aspect of sparsity to this problem that weak supervision can help with. Each question probably qualifies for more than five tags, but five is the maximum number of tags (in the data, 100 or so posts with six tags do show up) that can be assigned to any given question. This means the data is fairly sparse, almost like a standard multi-class problem if we seek maximum coverage with our labeler. We can use weak supervision to add labels to previously labeled posts to account for additional labels that were never an option.

![](README/stackoverflow_max_five_tags%204.png)

*Stack Overflow limits questions to a maximum of 5 tags, regardless of how many might really apply. Here I’m asking about implementing XML-CNN, an excellent algorithm for [extreme multilevel classification](http://manikvarma.org/downloads/XC/XMLRepository.html), and the UI prevents me from adding 8 tags that all apply.*

I like to ground my strategies in numbers, so let’s first look at the data. We’ll use pandas and numpy to model the data, but it is easier to use Spark to work with the raw data. Lets use PySpark to characterize the number of labels per question for the entire dataset:

```python
# Posts that have 1 answer, 1 net upvote and no parent posts
questions = spark.read.parquet(PATHS['questions'][PATH_SET])
questions.show(6)

+--------------------+--------------------+
|               _Body|               _Tags|
+--------------------+--------------------+
|Single Table Inhe...|<ruby-on-rails><r...|
|How can I use UIS...|<objective-c><uis...|
|WebDav client lib...|<iphone><objectiv...|
|Stagefright archi...|<android><stagefr...|
|byte code, librar...|    <java><bytecode>|
|JSP Progress Bar ...|<java><eclipse><j...|
+--------------------+--------------------+
```

```python
tags = questions.rdd.map(lambda x: re.sub('[<>]', ' ', x['_Tags']).split())
tags.take(3)

[['ruby-on-rails',
  'ruby',
  'ruby-on-rails-3',
  'class',
  'single-table-inheritance'],
 ['objective-c', 'uiscrollview', 'iphone'],
 ['iphone', 'objective-c', 'ios', 'webdav']]
```

```python
# Build a pyspark.sql.DataFrame and SQL table with schema: label/str, total/int
count_df = tags.groupBy(lambda x: len(x))\
    .map(
        lambda x: Row(label=x[0], total=len(x[1]))
    ).toDF()
count_df.registerTempTable("counts")

+-----------------+
|     average_tags|
+-----------------+
|3.065119621476553|
+-----------------+
```

We’ll create a DataFrame of tag counts to compute the average and plot the distribution of label counts.

```python
# Build a pyspark.sql.DataFrame and SQL table with schema: label/str, total/int
count_df = tags.groupBy(lambda x: len(x))\
    .map(
        lambda x: Row(label=x[0], total=len(x[1]))
    ).toDF()
count_df.registerTempTable("counts")

# Compute the average number of labels
spark.sql("SELECT ROUND(SUM(label * total)/SUM(total), 2) AS average_tags FROM counts").show()

+------------+
|average_tags|
+------------+
|        3.07|
+------------+
```

The [pyspark.sql.DataFrame.toPandas](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.toPandas) API is very handy for plotting PySpark DataFrames.

```python
%matplotlib inline
count_df.toPandas().plot(kind='bar', x='label', y='total', figsize=(4,6))
```

![](README/question_tag_distribution%204.png)

As you can see the distribution is fairly normal and symmetric, so both your average and median question has only three labels. If we can label more records for sparse tags using weak supervision, we should be able to expand the coverage of the model significantly.

## Balancing Act: Stratified Sampling
Unlike the Kaggle samples, in my early experiments with the raw Stack Overflow posts, I simply couldn’t get a tag classifier model to learn! The data was so imbalanced the model would optimize by learning to predict the most likely questions all the time, for every single record in the test data, and performance was terrible. Class and sample labels had little effect. I needed to resample the data.

There are several options for addressing imbalanced data:

* class weights - have the loss function apply more weight to **labels** of under-represented labels: 1/P(Y)
* sample weights - have the loss function apply more weight to **samples** with under-represented labels
* oversampling - sample records more frequently when their labels are under-represented, which may include duplicate records
* under sampling - sample records less often when their labels are over-represented, which will ignore some records
* synthetic data generation - create new records similar to records with under-represented labels

This is very simple for a binary or multi class classification problem. For multi-label problems, things are more complicated. It is necessary to implement an iterative technique, whereby the data is balanced given the current worst imbalance, it is re-measured, and this process is repeated. Such a method is outlined in [On the Stratification of Multi-Label Data](http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf) [Sechidis, Tsoumakas, Vlahavas, 2011]

### ETL in PySpark

Before we can work with the data we need to trim it down to size and change the format to something more efficient. We’ll be using the non-code text from the post initially, and the labels. 

#### Tools?

At first I used pandas for ETL but quickly ran into memory problems. Dask had the same issue or would freeze up, so Spark seemed the way to go.

### Parquet Format

The data comes in a series of files: as a single large XML document , and this is an incredibly slow format to compute with in Spark. First I uncompressed the data from 7zip format and then LZO compressed it for fast processing. Then I ran 

```python
posts_df = spark.read.format('xml').options(rowTag='row').options(rootTag='posts')\
                .load('data/stackoverflow/08-05-2019/Posts.xml.lzo')
posts_df.write.mode('overwrite')\
        .parquet('data/stackoverflow/08-05-2019/Posts.df.parquet')
```

## Building a Tag Classifier Model
We treat this as a multi-class, multi-label problem. 

The training data has been balanced by upsampling the complete dump of questions that have at least one answer, one vote and have at least one tag occurring more than 2,000 times. It is about 600MB. This dataset was [previously computed](https://github.com/rjurney/deep_products/blob/master/code/stackoverflow/get_questions.spark.py) and the files can be found in the *paas_blog/data* directory of the Github repo.

You can view the Jupyter Notebook with the code we’ll be running from Github at [github.com/rjurney/paas_blog/DCOS_Data_Science_Engine.ipynb](https://github.com/rjurney/paas_blog/blob/master/DCOS_Data_Science_Engine.ipynb). We’ll be opening it using the JupyterLab Github interface, but if you like you can paste its content block-by-block into a new Python 3 notebook.

### Loading the Tutorial Notebook

With Jupyter running (see , open the notebook called [deep_products/Weakly Supervised Learning - Stack Overflow Tag Labeler.ipynb](https://github.com/rjurney/deep_products/blob/master/code/Weakly%20Supervised%20Learning%20-%20Stack%20Overflow%20Tag%20Labeler.ipynb) and run it as you read along. 

### Verifying GPU Support

The first thing to do is to verify that our GPU is properly configured and working with Tensorflow 2.0. We use `tensorflow.test.is_gpu_available` and `tensorflow.compat.v2.config.experimental.list_physical_devices` to verify the GPUs are working with Tensorflow.

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

You can load the data for this tutorial using *pandas.read_parquet*. We supply a list of columns that is sorted 

```python
# Tag limit defines which dataset to load - those with tags having at least 50K, 20K, 10K, 5K or 2K instances
TAG_LIMIT = 2000

# Pre-computed sorted list of tag/index pairs
sorted_all_tags = json.load(open(f'data/stackoverflow/08-05-2019/sorted_all_tags.{TAG_LIMIT}.json'))
max_index = sorted_all_tags[-1][0] + 1

# Load the parquet file using pyarrow for this tag limit, using the sorted tag index to specify the columns
posts_df = pd.read_parquet(
    f'data/stackoverflow/08-05-2019/Questions.Stratified.Final.{TAG_LIMIT}.parquet',
    columns=['_Body'] + ['label_{}'.format(i) for i in range(0, max_index)],
    engine='pyarrow'
)
posts_df.head(2)
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
    '{:,} Stack Overflow questions with a tag having at least 2,000 occurrences'.format(
        len(posts_df.index)
    )
)
```

```
1,554,788 Stack Overflow questions with a tag having at least 2,000 occurrences
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

Note that we remove the *documents* and *sequences* variables and garbage collect, so as to conserve RAM.

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

We need one dataset to train with and one separate dataset to test and validate our model with.  The oft used [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) makes it so. Again we delete the previous variables to clear up space in RAM.

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

Although there has already been filtering and up-sampling of the data to restrict it to a sample of questions with at least one tag that occurs more than 2,000 times, there are still uneven ratios between common and uncommon labels. Without class weights, the most common label will be much more likely to be predicted than the least common. This resulted in the model being unable to learn anything useful. Class weights will make the loss function consider uncommon classes more than frequent ones, and the model can learn effectively.

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

![](README/kim_cnn_model_architecture%204.png)

<REMINDER: Ask Kim for permission to use this image before publishing!>

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

Metrics include names like *precision_66* which aren't consistent between runs. We fix these to cleanup our report on training the model.

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
```

We also add an f1 score, then make a DataFrame to display the log. This could be extended in repeat experiments.

```
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

y_pred = model.predict(X_test)
y_pred = (y_pred > CLASSIFY_THRESHOLD) * 1

y_pred_tags = []
for row in y_pred.tolist()[:TEST_COUNT]:
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

![](README/jupyter_results%204.png)


## Chapter Bibliography
* Liu, Chang, Wu, Yang, [Deep Learning for Extreme Multi-label Text Classification](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf), Carnegie Melon University, Pittsburgh, PA, USA, 2017


