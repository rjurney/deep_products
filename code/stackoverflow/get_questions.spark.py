import sys, os, re
import json

import boto3
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T


# spark = SparkSession.builder.appName('Deep Products - Sample JSON').getOrCreate()
# sc = spark.sparkContext

#
# Get answered questions and not their answers
#
posts = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Posts.df.parquet')
print('Total posts count: {:,}'.format(posts.count()))
questions = posts.filter(posts._ParentId.isNull())\
                 .filter(posts._AnswerCount > 0)
print('Total questions count: {:,}'.format(questions.count()))

# Write all questions to a Parquet file, then trim fields
questions\
    .write.mode('overwrite')\
    .parquet('s3://stackoverflow-events/08-05-2019/Questions.Answered.parquet')
questions = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Answered.parquet')

questions = questions.select('_Body', '_Tags')
questions.show()

# Count the number of each tag
all_tags = questions.rdd.flatMap(lambda x: re.sub('[<>]', ' ', x['_Tags']).split())

MAX_LEN = 100
PAD_TOKEN = '__PAD__'


def extract_text(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    tokens = doc.text.split()
    padded_tokens = [tokens[i] if len(tokens) > i else PAD_TOKEN for i in range(0, MAX_LEN)]
    return padded_tokens


for limit in [50000, 20000, 10000]:

    tag_counts_df = all_tags.groupBy(lambda x: x)\
        .map(lambda x: Row(tag=x[0], total=len(x[1])))\
        .toDF()\
        .select('tag', 'total')\
        .orderBy(['total'], ascending=False)
    tag_counts_df.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.TagCounts.{}.parquet'.format(limit))
    tag_counts_df = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.TagCounts.{}.parquet'.format(limit))
    tag_counts_df.show(100)

    local_tag_counts = tag_counts_df.rdd.collect()
    tag_counts = {x.tag:x.total for x in local_tag_counts}

    remaining_tags = tag_counts_df.filter(tag_counts_df.total > limit)
    total = remaining_tags.count()
    print('Tags with > {:,} instances: {:,}'.format(limit, total))

    top_tags = tag_counts_df.filter(tag_counts_df.total > limit)
    valid_tags = top_tags.rdd.map(lambda x: x['tag']).collect()

    # Turn text of body and tags into lists of words
    questions_lists = questions.rdd.map(
        lambda x: (
            extract_text(x['_Body']),
            re.sub('[<>]', ' ', x['_Tags']).split()
        )
    )

    # 1. Only questions with at least one tag in our list
    # 2. Drop tags not in our list
    filtered_lists = questions_lists.filter(
        lambda x: bool(set(x[1]) & set(valid_tags))            
    )\
        .map(lambda x: (x[0], [y for y in x[1] if y in valid_tags]))

    q_count = filtered_lists.count()
    print(
        'We are left with {:,} questions containing tags with over {:,} instances'.format(
            q_count,
            limit
        )
    )

    questions_tags = filtered_lists.map(lambda x: Row(_Body=x[0], _Tags=x[1])).toDF()
    questions_tags.show()
    questions_tags.select(
        '*', 
        F.size('_Tags').alias('_Tag_Count')
    ).orderBy(
        ['_Tag_Count'],
        ascending=False
    ).show()

    # Write the word/tag lists out
    questions_tags.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet'.format(limit))
    questions_tags = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet'.format(limit))

    # One-hot-encode the multilabel tags
    enumerated_labels = [z for z in enumerate(
        sorted(
            remaining_tags.rdd.groupBy(lambda x: 1)
                                .flatMap(lambda x: [y.tag for y in x[1]])
                                .collect()
        )
    )]
    tag_index = {x:i for i, x in enumerated_labels}
    index_tag = {i:x for i, x in enumerated_labels}

    def one_hot_encode(tag_list):
        """PySpark can't one-hot-encode multilabel data, so we do it ourselves."""

        one_hot_row = []
        for i, label in enumerated_labels:
            if index_tag[i] in tag_list:
                one_hot_row.append(1)
            else:
                one_hot_row.append(0)
        assert(len(one_hot_row) == len(enumerated_labels))
        return one_hot_row

    # Write the one-hot-encoded questions to S3 as a parquet file
    one_hot_questions = questions_tags.rdd.map(
        lambda x: Row(_Body=x._Body, _Tags=one_hot_encode(x._Tags))
    )
    one_hot_questions.take(10)

    # Verify we have multiple labels present
    one_hot_questions.sortBy(lambda x: sum(x._Tags), ascending=False).take(10)

    # Create a DataFrame for persisting as Parquet format
    schema = T.StructType([
        T.StructField("_Body", T.ArrayType(
            T.StringType()
        )),
        T.StructField("_Tags", T.ArrayType(
            T.IntegerType()
        ))
    ])

    one_hot_df = sqlContext.createDataFrame(
        one_hot_questions,
        schema
    )
    one_hot_df.show()
    one_hot_df.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.parquet'.format(limit))
    one_hot_df = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.parquet'.format(limit))

    def create_schema(one_row):
        schema_list = [
            T.StructField("_Body", T.ArrayType(
                T.StringType()
            )),
        ]

        for i, val in list(enumerate(one_row._Tags)):
            schema_list.append(
                T.StructField(
                    'label_{}'.format(i),
                    T.IntegerType()
                )
            )

        return T.StructType(schema_list)

    one_row = one_hot_df.take(1)[0]
    schema = create_schema(one_row)

    def create_row_columns(x):
        """Create a dict keyed with dynamic args to use to create a Row for this record"""
        args = {'label_{}'.format(i):val for i, val in list(enumerate(x._Tags))}
        args['_Body'] = x._Body
        return Row(**args)

    stratify_limit = 10000

    output_rdd = sc.emptyRDD()
    for i in range(0, len(one_row._Tags)):
        positive_examples = one_hot_df.rdd.filter(lambda x: x._Tags[i])
        example_count = positive_examples.count()
        ratio = min(1.0, stratify_limit / example_count)
        sample_ratio = max(0.0, ratio)
        positive_examples = positive_examples.sample(False, sample_ratio, seed=1337).map(create_row_columns)
        sample_count = positive_examples.count()
        print(
            'Column {:,} had {:,} positive examples, sampled to {:,}'.format(
                i,
                example_count, sample_count
            )
        )
        output_df = sqlContext.createDataFrame(
            positive_examples,
            schema
        )
        output_df.show()
        output_df.write.mode('overwrite').json('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.{}.jsonl'.format(limit, i))

    #
    # Store the associated files to S3 as JSON
    #
    s3 = boto3.resource('s3')

    obj = s3.Object('stackoverflow-events', '08-05-2019/tag_index.{}.json'.format(limit))
    obj.put(Body=json.dumps(tag_index).encode())

    obj = s3.Object('stackoverflow-events', '08-05-2019/index_tag.{}.json'.format(limit))
    obj.put(Body=json.dumps(index_tag).encode())

    obj = s3.Object('stackoverflow-events', '08-05-2019/sorted_all_tags.{}.json'.format(limit))
    obj.put(Body=json.dumps(enumerated_labels).encode())

    # Evaluate how skewed the sample is
    stratified_sample = spark.read.json('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.*.jsonl'.format(limit))
    stratified_sample.registerTempTable('stratified_sample')

    label_counts = {}
    for i in range(0, 100):
        count_df = spark.sql('SELECT label_{}, COUNT(*) as total FROM stratified_sample GROUP BY label_{}'.format(i, i))
        rows = count_df.rdd.take(2)
        neg_count = getattr(rows[0], 'total')
        pos_count = getattr(rows[1], 'total')
        label_counts[i] = [neg_count, pos_count]

    # Put the label counts on S3
    obj = s3.Object('stackoverflow-events', '08-05-2019/label_counts.{}.json'.format(limit))
    obj.put(Body=json.dumps(label_counts).encode())

    # Write the final stratified sample to Parquet format
    stratified_sample.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.Final.{}.parquet'.format(limit))

# Compute a report on the dupes in the data
for limit in [50000, 20000, 10000]:
    data = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.Final.{}.parquet'.format(limit))
    data.registerTempTable("data")
    raw_total = data.count()
    df = spark.sql(
        """SELECT COUNT(*) as total FROM (SELECT DISTINCT {} FROM data)""".format(
            ', '.join(data.columns[1:])[0:-2]
        )
    )
    unique_total = df.rdd.first().total
    dupe_total = raw_total - unique_total
    print('Limit {} has {} total, {} unique and {} duplicate labelsets'.format(
        raw_total,
        unique_total,
        dupe_total
    ))

# Compute the remaining tag counts
# for limit in [50000, 20000, 10000]:
#     questions_tags = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet'.format(limit))


# # questions = questions.limit(1000000)
# # questions\
# #     .write.mode('overwrite')\
# #     .parquet('s3://stackoverflow-events/07-19-2019/Questions.Answered.1M.parquet')
# # sample_posts = spark.read.parquet('s3://stackoverflow-events/07-19-2019/Questions.Answered.1M.parquet')



# # Write the Q&A post IDs for later joins
# sample_qa_ids = sample_posts.select('_Id').withColumnRenamed('_Id', '_SId')
# sample_qa_ids\
#     .coalesce(1)\
#     .write.mode('overwrite')\
#     .json('s3://stackoverflow-events/06-24-2019/SampleIds.100K.Questions.jsonl.gz',
#           compression='gzip')
# sample_qa_ids = spark.read.json('s3://stackoverflow-events/06-24-2019/SampleIds.100K.Questions.jsonl.gz')


# #
# # Get the corresponding comments
# #
# comments = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Comments.df.parquet')
# sample_comments = comments.join(sample_qa_ids,
#                                 sample_qa_ids._SId == comments._PostId)
# print('Total sample comments: {:,}'.format(sample_comments.count()))
# sample_comments.drop('_SId')\
#     .write.mode('overwrite')\
#     .json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz',
#           compression='gzip')
# sample_comments = spark.read.json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz')

# #
# # Get users participating in those posts
# #
# users = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Users.df.parquet')\
#              .alias('Users')
# sample_posts_users = sample_posts.join(users,
#                                        sample_posts._OwnerUserId == users._Id)
# print('Full users sample: {:,}'.format(sample_posts_users.count()))
# sample_users = sample_posts_users.select(['Users.' + c for c in users.columns]).distinct()
# print('Distinct users sample: {:,}'.format(sample_users.count()))
# sample_users.write.mode('overwrite')\
#             .json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz',
#                   compression='gzip')
# sample_users = spark.read.json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz')

# #
# # Get votes on those posts
# #
# votes = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Votes.df.parquet')\
#              .alias('Votes')
# sample_posts_votes = votes.join(sample_qa_ids,
#                                 sample_qa_ids._SId == votes._PostId)
# print('Full votes sample: {:,}'.format(sample_posts_votes.count()))
# sample_votes = sample_posts_votes.drop('_SId')
# sample_votes.write.mode('overwrite')\
#             .json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz',
#                   compression='gzip')
# sample_votes = spark.read.json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz')

# #
# # Get the corresponding post histories
# #
# post_history = spark.read.parquet('s3://stackoverflow-events/06-24-2019/PostHistory.df.parquet')\
#                     .alias('PostHistory')
# sample_post_history = post_history.join(sample_qa_ids,
#                                         sample_qa_ids._SId == post_history._PostId)
# print('Full post history sample: {:,}'.format(sample_post_history.count()))
# sample_post_history.drop('_SId')\
#     .write.mode('overwrite')\
#     .json('s3://stackoverflow-events/06-24-2019/PostHistory.100K.Questions.jsonl.gz',
#           compression='gzip')
# sample_post_history = spark.read.json('s3://stackoverflow-events/06-24-2019/PostHistory.100K.Questions.jsonl.gz')

# #
# # Get the corresponding badges
# #
# badges = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Badges.df.parquet')
# user_ids = sample_users.select('_Id')\
#                        .withColumn('_UId', sample_users._Id)\
#                        .drop('_Id')\
#                        .distinct()
# sample_badges = badges.join(user_ids,
#                             badges._UserId == user_ids._UId).drop('_UId')
# print('Full badges sample: {:,}'.format(sample_badges.count()))
# sample_badges.write.mode('overwrite')\
#     .json('s3://stackoverflow-events/06-24-2019/Badges.100K.Questions.jsonl.gz',
#           compression='gzip')
# sample_badges = spark.read.json('s3://stackoverflow-events/06-24-2019/Badges.100K.Questions.jsonl.gz')

# # Leave tags alone, tiny
# # tags = spark.read.parquet('data/stackoverflow/parquet/Tags.df.parquet')
