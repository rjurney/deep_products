import gc
import json
import re

import boto3
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Toeknize questions, remove stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Print debug info as we compute, takes extra time
DEBUG = False

# Print a report on record/label duplication at the end
REPORT = True

# Define a set of paths for each step for local and S3
PATH_SET = 'local'

PATHS = {
    's3_bucket': 'stackoverflow-events',
    'posts': {
        'local': 'data/stackoverflow/08-05-2019/Posts.df.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Posts.df.parquet',
    },
    'questions': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Answered.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Answered.parquet',
    },
    'tag_counts': {
        'local': 'data/stackoverflow/08-05-2019/Questions.TagCounts.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.TagCounts.{}.parquet',
    },
    'questions_tags': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Tags.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet',
    },
    'one_hot': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Stratified.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.parquet',
    },
    'output_jsonl': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Stratified.{}.{}.jsonl',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.{}.jsonl',
    },
    'tag_index': {
        'local': 'data/stackoverflow/08-05-2019/tag_index.{}.json',
        's3': '08-05-2019/tag_index.{}.json',
    },
    'index_tag': {
        'local': 'data/stackoverflow/08-05-2019/index_tag.{}.json',
        's3': '08-05-2019/index_tag.{}.json',
    },
    'sorted_all_tags': {
        'local': 'data/stackoverflow/08-05-2019/sorted_all_tags.{}.json',
        's3': '08-05-2019/sorted_all_tags.{}.json',
    },
    'stratified_sample': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Stratified.{}.*.jsonl',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Stratified.{}.*.jsonl',
    },
    'label_counts': {
        'local': 'data/stackoverflow/08-05-2019/label_counts.{}.json',
        's3': '08-05-2019/label_counts.{}.json',
    },
    'questions_final': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Stratified.Final.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Stratified.Final.{}.parquet'
    },
    'report': {
        'local': 'data/stackoverflow/08-05-2019/final_report.{}.json',
        's3': '08-05-2019/final_report.{}.json',
    },
    'bad_questions': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Bad.{}.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Bad.{}.{}.parquet',
    }
}


#
# Initialize Spark with dynamic allocation enabled to (hopefully) use less RAM
#
spark = SparkSession.builder\
    .appName('Deep Products - Sample JSON')\
    .config('spark.dynamicAllocation.enabled', True)\
    .config('spark.shuffle.service.enabled', True)\
    .getOrCreate()
sc = spark.sparkContext


#
# Get answered questions and not their answers
#
posts = spark.read.parquet(PATHS['posts'][PATH_SET])
if DEBUG is True:
    print('Total posts count: {:,}'.format(
        posts.count()
    ))
questions = posts.filter(posts._ParentId.isNull())\
                 .filter(posts._AnswerCount > 0)\
                 .filter(posts._Score > 1)
if DEBUG is True:
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
    .parquet(PATHS['questions'][PATH_SET])
questions = spark.read.parquet(PATHS['questions'][PATH_SET])

#
# Split off questions with no tags for weak supervision
#
untagged_questions = questions.filter(F.length('_Tags') == 0)

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


# Prepare multiple datasets with different tag count frequency filters and per-tag
# stratified sample sizes
for tag_limit, stratify_limit, lower_limit in \
    [
        (50000, 50000, 500),
        (20000, 10000, 500),
        (10000, 10000, 500),
        (5000, 5000, 500),
        (2000, 2000, 500),
        (1000, 1000, 500),
    ]:

    print(f'\n\nStarting run for tag limit {tag_limit:,} and sample size {stratify_limit:,}\n\n')

    tag_counts_df = all_tags\
        .groupBy(lambda x: x)\
        .map(lambda x: Row(tag=x[0], total=len(x[1])))\
        .toDF()\
        .select('tag', 'total').orderBy(['total'], ascending=False)
    tag_counts_df.write.mode('overwrite').parquet(PATHS['tag_counts'][PATH_SET].format(tag_limit))
    tag_counts_df = spark.read.parquet(PATHS['tag_counts'][PATH_SET].format(tag_limit))

    if DEBUG is True:
        tag_counts_df.show(100)

    local_tag_counts = tag_counts_df.rdd.collect()
    tag_counts = {x.tag: x.total for x in local_tag_counts}

    remaining_tags = tag_counts_df.filter(tag_counts_df.total > tag_limit)
    bad_tags       = tag_counts_df.filter(
        tag_counts_df.total <= tag_limit & tag_counts_df.total >= lower_limit
    )
    tag_total = remaining_tags.count()
    bad_tag_total = bad_tags.count()
    print(f'\n\nNumber of tags with > {tag_limit:,} instances: {tag_total:,}')
    print(f'Number of tags with >= {lower_limit:,} and lower than/equal to {tag_limit:,}\n\n')

    top_tags = tag_counts_df.filter(tag_counts_df.total > tag_limit)
    valid_tags = top_tags.rdd.map(lambda x: x['tag']).collect()

    # Turn text of body and tags into lists of words
    questions_lists = questions.rdd.map(
        lambda x: (extract_text(x['_Body']), re.sub('[<>]', ' ', x['_Tags']).split())
    )

    # 1. Only questions with at least one tag in our list
    # 2. Drop tags not in our list
    filtered_lists = questions_lists\
        .filter(lambda x: bool(set(x[1]) & set(valid_tags)))\
        .map(lambda x: (x[0], [y for y in x[1] if y in valid_tags]))

    # Set aside other questions without frequent enough tags for enrichment via Snorkel
    bad_questions = questions_lists\
        .filter(lambda x: bool(set(x[1]) & set(bad_tags)))\
        .map(lambda x: (x[0], [y for y in x[1] if y in bad_tags]))
    bad_questions_df = bad_questions.map(lambda x: Row(_Body=x[0], _Tags=x[1])).toDF()
    bad_questions_df.write.mode('overwrite').parquet(
        PATHS['bad_questions'][PATH_SET].format(tag_limit, lower_limit)
    )

    if DEBUG is True:
        q_count = filtered_lists.count()
        print(f'\n\nWe are left with {q_count:,} questions containing tags with over {tag_limit:,} instances\n\n')

    questions_tags = filtered_lists.map(lambda x: Row(_Body=x[0], _Tags=x[1])).toDF()
    if DEBUG is True:
        questions_tags.show()

    # Write the word/tag lists out
    questions_tags.write.mode('overwrite').parquet(path['questions_tags'][PATH_SET].format(tag_limit))
    questions_tags = spark.read.parquet(path['questions_tags'][PATH_SET].format(tag_limit))

    # One-hot-encode the multilabel tags
    enumerated_labels = [
        z for z in enumerate(
            sorted(
                remaining_tags.rdd
                .groupBy(lambda x: 1)
                .flatMap(lambda x: [y.tag for y in x[1]])
                .collect()
            )
        )
    ]
    tag_index = {x: i for i, x in enumerated_labels}
    index_tag = {i: x for i, x in enumerated_labels}

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
    if DEBUG is True:
        print(
            one_hot_questions.take(10)
        )
        # Verify we have multiple labels present
        print(
            one_hot_questions.sortBy(lambda x: sum(x._Tags), ascending=False).take(10)
        )

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
    one_hot_df.write.mode('overwrite').parquet(PATHS['one_hot'][PATH_SET])
    one_hot_df = spark.read.parquet(PATHS['one_hot'][PATH_SET])

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
        args = {'label_{}'.format(i): val for i, val in list(enumerate(x._Tags))}
        args['_Body'] = x._Body
        return Row(**args)

    for i in range(0, tag_total):
        print(f'\n\nProcessing tag limit: {tag_limit:,} stratify limit: {stratify_limit:,} tag {i:,} of {tag_total:,} total tags\n\n')
        positive_examples = one_hot_df.rdd.filter(lambda x: x._Tags[i])
        example_count = positive_examples.count()
        ratio = min(1.0, stratify_limit / example_count)
        sample_ratio = max(0.0, ratio)
        positive_examples = positive_examples.sample(False, sample_ratio, seed=1337).map(create_row_columns)

        if DEBUG is True:
            sample_count = positive_examples.count()
            print(
                f'Column {i:,} had {example_count:,} positive examples, sampled to {sample_count:,}'
            )

        output_df = spark.createDataFrame(
            positive_examples,
            schema
        )

        if DEBUG is True:
            output_df.show()

        output_df.write.mode('overwrite').json(PATHS['output_jsonl'][PATH_SET].format(tag_limit, i))

        # Avoid RAM problems
        del output_df
        gc.collect()

#
# Store the associated files to local disk or S3 as JSON
#
s3 = boto3.resource('s3')

if PATH_SET == 's3':
    obj = s3.Object(PATHS['s3_bucket'], PATHS['tag_index']['s3'].format(tag_limit))
    obj.put(Body=json.dumps(tag_index).encode())

    obj = s3.Object(PATHS['s3_bucket'], PATHS['index_tag']['s3'].format(tag_limit))
    obj.put(Body=json.dumps(index_tag).encode())

    obj = s3.Object(PATHS['s3_bucket'], PATHS['sorted_all_tags']['s3'].format(tag_limit))
    obj.put(Body=json.dumps(enumerated_labels).encode())
else:
    json.dump(tag_index, open(PATHS['tag_index']['local'].format(tag_limit), 'w'))
    json.dump(tag_index, open(PATHS['index_tag']['local'].format(tag_limit), 'w'))
    json.dump(tag_index, open(PATHS['sorted_all_tags']['local'].format(tag_limit), 'w'))


# Evaluate how skewed the sample is
stratified_sample = spark.read.json(PATHS['stratified_sample'][PATH_SET].format(tag_limit))
stratified_sample.registerTempTable('stratified_sample')

label_counts = {}

# I wish this could be optimized but I don't know how...
for i in range(0, tag_total):
    count_df = spark.sql(f'SELECT label_{i}, COUNT(*) as total FROM stratified_sample GROUP BY label_{i}')
    rows = count_df.rdd.take(2)
    neg_count = getattr(rows[0], 'total')
    pos_count = getattr(rows[1], 'total')
    label_counts[i] = [neg_count, pos_count]

    # Manage memory explicitly to avoid out of RAM errors
    del count_df
    gc.collect()

# Put the label counts on local disk or S3
if PATH_SET == 's3':
    obj = s3.Object(PATHS['s3_bucket'], PATHS['label_counts']['s3'].format(tag_limit))
    obj.put(Body=json.dumps(label_counts).encode())
else:
    json.dumps(label_counts, open(PATHS['label_counts']['local'].format(tag_limit), 'w'))

# Write the final stratified sample to Parquet format
stratified_sample.write.mode('overwrite').parquet(PATHS['questions_final'][PATH_SET].format(tag_limit))
stratified_sample = spark.read.parquet(PATHS['questions_final'][PATH_SET].format(tag_limit))

# Blow away the old stratified sample table
spark.catalog.dropTempView("stratified_sample")

#
# Compute a report on the data
#

if REPORT is True:

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
    print('Limit {tag_limit:,} has {raw_total:,} total, {unique_total:,} unique and {dupe_total:,} duplicate labelsets with a dupe ratio of {dupe_ratio:,}')

    one_hot_original = one_hot_df.rdd.map(create_row_columns)
    original_df = spark.createDataFrame(one_hot_original, schema)
    original_df.registerTempTable("original_data")

    original_raw_total = original_df.count()
    select_cols = ', '.join(original_df.columns[1:])
    original_report_df = spark.sql(
        f"SELECT COUNT(*) as total FROM (SELECT DISTINCT {select_cols} FROM original_data)"
    )
    original_unique_total = original_report_df.rdd.first().total
    original_dupe_total = original_raw_total - unique_total
    original_dupe_ratio = original_dupe_total * 1.0 / original_raw_total * 1.0

    # Print and store a report on duplicates in the original
    print(f'Limit {tag_limit:,} originally had {original_raw_total:,} total, {original_unique_total:,} unique and {original_dupe_total:,} duplicate labelsets with a dupe ratio of {original_dupe_ratio:,}')

    dupe_ratio_change = original_dupe_ratio - dupe_ratio
    dupe_ratio_change_pct = dupe_ratio / original_dupe_ratio

    print(f'Dupe ratio change raw/pct: {dupe_ratio_change:,}/{dupe_ratio_change_pct:,}')

    report_data = {'raw_total': raw_total, 'unique_total': unique_total, 'dupe_total': dupe_total, 'dupe_ratio': dupe_ratio, 'original_raw_total': original_raw_total, 'original_unique_total': original_unique_total, 'original_dupe_total': original_dupe_total, 'original_dupe_ratio': original_dupe_ratio, 'dupe_ratio_change': dupe_ratio_change, 'dupe_ratio_change_pct': dupe_ratio_change_pct}

    # Write the report to local disk or S3
    if PATH_SET == 's3':
        obj = s3.Object(PATHS['s3_bucket'], PATHS['report']['s3'].format(tag_limit))
        obj.put(Body=json.dumps(report_data).encode())
    else:
        json.dump(report_data, open(PATHS['report']['local'].format(tag_limit), 'w'))

    # The Big Finish(TM)!
