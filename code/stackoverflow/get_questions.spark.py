import sys, os, re
import json

import boto3
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Remove stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


spark = SparkSession.builder.appName('Deep Products - Sample JSON').getOrCreate()
sc = spark.sparkContext

#
# Get answered questions and not their answers
#
posts = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Posts.df.parquet')
print('Total posts count: {:,}'.format(posts.count()))
questions = posts.filter(posts._ParentId.isNull())\
                 .filter(posts._AnswerCount > 0)\
                 .filter(posts._Score > 1)
print('Total questions count: {:,}'.format(questions.count()))

questions = questions.select(
    F.concat(
        F.col("_Title"),
        F.lit(" "),
        F.col("_Body")
    ).alias('_Body'),
    '_Tags'
)
questions.show()

# Write all questions to a Parquet file, then trim fields
questions\
    .write.mode('overwrite')\
    .parquet('s3://stackoverflow-events/08-05-2019/Questions.Answered.parquet')
questions = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Answered.parquet')

# Count the number of each tag
all_tags = questions.rdd.flatMap(lambda x: re.sub('[<>]', ' ', x['_Tags']).split())

MAX_LEN = 200
PAD_TOKEN = '__PAD__'
tokenizer = RegexpTokenizer(r'\w+')

def extract_text(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    text = re.sub(r'http\S+', ' ', doc.text)
    tokens = [x for x in tokenizer.tokenize(text) if x not in stop_words]

    padded_tokens = [tokens[i] if len(tokens) > i else PAD_TOKEN for i in range(0, MAX_LEN)]
    return padded_tokens


# Prepare multiple datasets with different tag count frequency filters and per-tag stratified sample sizes
for tag_limit, stratify_limit in [
        (50000, 50000),
        (20000, 10000),
        (10000, 10000),
        (5000, 5000),
        (2000, 2000),
        (1000, 1000)
    ]:

    print(f'\n\nStarting run for tag limit {tag_limit:,} and sample size {stratify_limit:,}\n\n')

    tag_counts_df = all_tags.groupBy(lambda x: x)\
        .map(lambda x: Row(tag=x[0], total=len(x[1])))\
        .toDF()\
        .select('tag', 'total')\
        .orderBy(['total'], ascending=False)
    tag_counts_df.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.TagCounts.{}.parquet'.format(tag_limit))
    tag_counts_df = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.TagCounts.{}.parquet'.format(tag_limit))
    tag_counts_df.show(100)

    local_tag_counts = tag_counts_df.rdd.collect()
    tag_counts = {x.tag:x.total for x in local_tag_counts}

    remaining_tags = tag_counts_df.filter(tag_counts_df.total > tag_limit)
    total = remaining_tags.count()

    print(f'\n\nNumber of tags with > {tag_limit:,} instances: {total:,}\n\n')

    top_tags = tag_counts_df.filter(tag_counts_df.total > tag_limit)
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
    print(f'\n\nWe are left with {q_count:,} questions containing tags with over {tag_limit:,} instances\n\n')

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
    questions_tags.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet'.format(tag_limit))
    questions_tags = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet'.format(tag_limit))

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

    def one_hot_encode(tag_list, enumerated_labels):
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
        lambda x: Row(_Body=x._Body, _Tags=one_hot_encode(x._Tags, enumerated_labels))
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

    one_hot_df = spark.createDataFrame(
        one_hot_questions,
        schema
    )
    one_hot_df.show()
    one_hot_df.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.parquet'.format(tag_limit))
    one_hot_df = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.parquet'.format(tag_limit))

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

    output_rdd = sc.emptyRDD()
    row_tag_count = len(one_row._Tags)
    for i in range(0, row_tag_count):

        print(f'\n\nProcessing tag limit: {tag_limit:,} stratify limit: {stratify_limit:,} tag {i:,} of {row_tag_count:,} total tags\n\n')

        positive_examples = one_hot_df.rdd.filter(lambda x: x._Tags[i])
        example_count = positive_examples.count()
        ratio = min(1.0, stratify_limit / example_count)
        sample_ratio = max(0.0, ratio)
        positive_examples = positive_examples.sample(False, sample_ratio, seed=1337).map(create_row_columns)
        sample_count = positive_examples.count()
        print(
            f'Column {i:,} had {example_count:,} positive examples, sampled to {sample_count:,}'
        )
        output_df = spark.createDataFrame(
            positive_examples,
            schema
        )
        output_df.show()
        output_df.write.mode('overwrite').json('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.{}.jsonl'.format(tag_limit, i))

    #
    # Store the associated files to S3 as JSON
    #
    s3 = boto3.resource('s3')

    obj = s3.Object('stackoverflow-events', '08-05-2019/tag_index.{}.json'.format(tag_limit))
    obj.put(Body=json.dumps(tag_index).encode())

    obj = s3.Object('stackoverflow-events', '08-05-2019/index_tag.{}.json'.format(tag_limit))
    obj.put(Body=json.dumps(index_tag).encode())

    obj = s3.Object('stackoverflow-events', '08-05-2019/sorted_all_tags.{}.json'.format(tag_limit))
    obj.put(Body=json.dumps(enumerated_labels).encode())

    # Count the labels
    label_count = len(enumerated_labels)

    # Evaluate how skewed the sample is
    stratified_sample = spark.read.json('s3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.*.jsonl'.format(tag_limit))
    stratified_sample.registerTempTable('stratified_sample')

    label_counts = {}
    for i in range(0, label_count):
        count_df = spark.sql('SELECT label_{}, COUNT(*) as total FROM stratified_sample GROUP BY label_{}'.format(i, i))
        rows = count_df.rdd.take(2)
        neg_count = getattr(rows[0], 'total')
        pos_count = getattr(rows[1], 'total')
        label_counts[i] = [neg_count, pos_count]

    # Put the label counts on S3
    obj = s3.Object('stackoverflow-events', '08-05-2019/label_counts.{}.json'.format(tag_limit))
    obj.put(Body=json.dumps(label_counts).encode())

    # Write the final stratified sample to Parquet format
    stratified_sample.write.mode('overwrite').parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.Final.{}.parquet'.format(tag_limit))
    stratified_sample = spark.read.parquet('s3://stackoverflow-events/08-05-2019/Questions.Stratified.Final.{}.parquet'.format(tag_limit))

    # Blow away the old stratified sample table
    spark.catalog.dropTempView("stratified_sample")

    # Register a new table to compute duplicate ratios
    stratified_sample.registerTempTable("final_stratified_sample")
    raw_total = stratified_sample.count()
    report_df = spark.sql(
        """SELECT COUNT(*) as total FROM (SELECT DISTINCT {} FROM final_stratified_sample)""".format(
            ', '.join(stratified_sample.columns[1:])
        )
    )
    unique_total = report_df.rdd.first().total
    dupe_total = raw_total - unique_total
    dupe_ratio = dupe_total * 1.0 / raw_total * 1.0

    # Print and store a report on duplicates in the sample
    print('Limit {:,} has {:,} total, {:,} unique and {:,} duplicate labelsets with a dupe ratio of {:,}'.format(
        tag_limit,
        raw_total,
        unique_total,
        dupe_total,
        dupe_ratio
    ))

    one_hot_original = one_hot_df.rdd.map(create_row_columns)
    original_df = spark.createDataFrame(
        one_hot_original,
        schema
    )
    original_df.registerTempTable("original_data")

    original_raw_total = original_df.count()
    original_report_df = spark.sql(
        """SELECT COUNT(*) as total FROM (SELECT DISTINCT {} FROM original_data)""".format(
            ', '.join(original_df.columns[1:])
        )
    )
    original_unique_total = original_report_df.rdd.first().total
    original_dupe_total = original_raw_total - unique_total
    original_dupe_ratio = original_dupe_total * 1.0 / original_raw_total * 1.0

    # Print and store a report on duplicates in the original
    print('Limit {:,} originally had {:,} total, {:,} unique and {:,} duplicate labelsets with a dupe ratio of {:,}'.format(
        tag_limit,
        original_raw_total,
        original_unique_total,
        original_dupe_total,
        original_dupe_ratio
    ))

    dupe_ratio_change = original_dupe_ratio - dupe_ratio
    dupe_ratio_change_pct = dupe_ratio / original_dupe_ratio

    print('Dupe ratio change raw/pct: {:,}/{:,}'.format(dupe_ratio_change, dupe_ratio_change_pct))

    report_data = {
        'raw_total': raw_total,
        'unique_total': unique_total,
        'dupe_total': dupe_total,
        'dupe_ratio': dupe_ratio,
        'original_raw_total': original_raw_total,
        'original_unique_total': original_unique_total,
        'original_dupe_total': original_dupe_total,
        'original_dupe_ratio': original_dupe_ratio,
        'dupe_ratio_change': dupe_ratio_change,
        'dupe_ratio_change_pct': dupe_ratio_change_pct
    }
    obj = s3.Object('stackoverflow-events', '08-05-2019/final_report.{}.json'.format(tag_limit))
    obj.put(Body=json.dumps(report_data).encode())

# The Big Finish(TM)!
