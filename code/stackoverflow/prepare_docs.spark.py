from frozendict import frozendict
import json
from pyspark.sql import SparkSession, Row

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
post_comments = posts_df.join(comments_df, posts_df._Id == comments_df._PostId, how='left_outer')
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

os.system('aws s3 rm --recursive s3://stackoverflow-events/06-24-2019/combined/Posts.Comments.jsonl.gz')
combined_docs.map(lambda x: json.dumps(x)).coalesce(1).saveAsTextFile(
    path='s3://stackoverflow-events/06-24-2019/combined/Posts.Comments.jsonl.gz',
    compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec"
)

print('Original docs: {:,}'.format(combined_docs.count()))
print('Docs with _OwnerUserId: {:,}'.format(
    combined_docs.filter(lambda x: '_OwnerUserId' in x and x['_OwnerUserId'] is not None).count())
)

filtered_docs = combined_docs.filter(lambda x: '_OwnerUserId' in x and x['_OwnerUserId'] is not None)
print('Step 1 docs: {:,}'.format(filtered_docs.count()))

joinable_docs = filtered_docs.map(lambda x: (x['_OwnerUserId'], x))\
                             .filter(lambda x: x[0] is not None)
print('Step 2 docs: {:,}'.format(joinable_docs.count()))

users_df = spark.read.json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz')
users = users_df.rdd.filter(lambda x: x['_Id'] is not None)
print('Original docs: {:,}'.format(users_df.count()))
print('With ID: {:,}'.format(users.count()))

def choose_list(x):
    if not x or len(x) < 1:
        print("E")
        return []
    return x

users_docs = joinable_docs.leftOuterJoin(
    users.filter(lambda x: '_Id' in x and x['_Id'] is not None).map(lambda x: (x['_Id'], x))
).map(lambda x: x[1]).map(lambda x: (x[0], x[1]))
print('user_docs: {:,}'.format(users_docs.count()))

def update_owner(x):
    post = x[0]
    users = x[1]
    if users is None or users == '':
        users = None

    if isinstance(users, Row):
        users = users.asDict()

    if isinstance(users, str):
        users = None

    post.update({'_Owner': users})
    return post
updated_users_docs = users_docs.map(update_owner)
print('Total updated_users_docs: {:,}'.format(updated_users_docs.count()))

joinable_docs = updated_users_docs.map(lambda x: (x['_Id'], x))

votes_df = spark.read.json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz')
joinable_votes = votes_df.rdd.filter(lambda x: '_PostId' in x and x['_PostId'] is not None).map(lambda x: (x['_PostId'], x))
joinable_votes.count()

vote_groups = votes_df.rdd.groupBy(lambda x: x['_PostId']).map(lambda x: (x[0], [y.asDict() for y in x[1]]))

docs_votes_joined = joinable_docs.leftOuterJoin(vote_groups).map(lambda x: x[1])

def update_vote(x):
    post = x[0]
    votes = x[1]

    post.update({'_Votes': votes})
    return post
docs_votes = docs_votes_joined.map(update_vote)
print('docs_votes count: {:,}'.format(docs_votes.count()))


os.system('aws s3 rm --recursive s3://stackoverflow-events/06-24-2019/CombinedDocs.100K.Questions.jsonl.gz')
docs_votes.