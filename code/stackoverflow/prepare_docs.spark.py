from frozendict import frozendict
import json
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Deep Products - Convert XML to Parquet').getOrCreate()
sc = spark.sparkContext


posts_df = spark.read.json('s3://stackoverflow-events/06-24-2019/Posts.100K.Questions.jsonl.gz').alias('Posts')
print('Total posts: {:,}'.format(posts_df.count()))

comments_df = spark.read.json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz').alias('Comments')
print('Total comments: {:,}'.format(comments_df.count()))

# Evaluate for nulls
print('Non null posts count: {:,}'.format(posts_df.filter(posts_df._Id.isNotNull()).count()))
print('Non null comments count: {:,}'.format(comments_df.filter(comments_df._PostId.isNotNull()).count()))

print([x['_Id'] for x in posts_df.orderBy('_Id').take(10)])
print([x['_PostId'] for x in comments_df.orderBy('_PostId').take(10)])


# Join and evaluate
post_comments = posts_df.join(comments_df, posts_df._Id == comments_df._PostId)
print('Post comments count: {:,}'.format(post_comments.count()))

comment_columns = [col for col in comments_df.columns]
post_columns = [col for col in posts_df.columns]

post_comment_dicts = \
    post_comments.rdd.map(
    lambda t: (
        frozendict({col:t[col] for col in post_columns}), 
        [{col:t[col] for col in comment_columns}]
    )
)
print('Post comments dict count: {:,}'.format(post_comment_dicts.count()))


def compose_dict(x):
    """Compose a final dictionary of a record with the Post fields present and the Comment records under 
       the 'Comments' key"""
    post = dict(x[0])
    comments = sorted(x[1], key=lambda y: y['_CreationDate'])
    post.update({'Comments': comments})
    return post

combined_docs = post_comment_dicts.reduceByKey(lambda a, b: a + b).map(compose_dict)
print('Combined docs count: {:,}'.format(combined_docs.count()))

os.system('rm -rf data/stackoverflow/sample/combined/Posts.Comments.jsonl.gz')
combined_docs.map(lambda x: json.dumps(x)).coalesce(1).saveAsTextFile(
    path='data/stackoverflow/sample/combined/Posts.Comments.jsonl.gz',
    compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec"
)