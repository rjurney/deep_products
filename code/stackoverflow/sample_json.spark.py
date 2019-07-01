from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder.appName('Deep Products - Sample JSON').getOrCreate()
sc = spark.sparkContext


#
# Get 100K answered questions and their answers
#

posts = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Posts.df.parquet')
# posts.select('_ParentId', '_Body').filter(posts._ParentId == 9915705).show()

# Question IDs with answers - 14,944,519
answered_question_ids = posts.filter(posts._ParentId.isNull())\
                             .filter(posts._AnswerCount > 0)\
                             .select(posts._Id.alias('_QId'))\
                             .distinct()
sample_ratio = 100000.0 / answered_question_ids.count()  # 0.006691416431669698
sample_question_ids = answered_question_ids.sample(False, sample_ratio, 1101)

questions = posts.join(F.broadcast(sample_question_ids),
                       sample_question_ids._QId == posts._Id)
assert(questions.count() == sample_question_ids.count())

answers = posts.join(F.broadcast(sample_question_ids),
                     sample_question_ids._QId == posts._ParentId)
assert(answers.select('_ParentId').distinct().count() == sample_question_ids.count())

sample_q_and_a = questions.union(answers)
sample_posts = sample_q_and_a.drop('_QId')
# assert(sample_posts.count() == questions.count() + answers.count())

# Write to a single JSON Lines file (which will be within the directory specified below)
sample_posts\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/Posts.100K.Questions.jsonl.gz',
          compression='gzip')

# Write the Q&A post IDs for later joins
sample_qa_ids = sample_posts.select('_Id').withColumnRenamed('_Id', '_SId')
sample_qa_ids\
    .coalesce(1)\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/SampleIds.100K.Questions.jsonl.gz',
          compression='gzip')


#
# Get the corresponding comments
#
comments = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Comments.df.parquet')
sample_comments = comments.join(F.broadcast(sample_qa_ids),
                                sample_qa_ids._SId == comments._PostId)
sample_comments.drop('_SId')\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz',
          compression='gzip')


#
# Get users participating in those posts
#
users = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Users.df.parquet')\
             .alias('Users')
sample_posts_users = sample_posts.join(users,
                                       sample_posts._OwnerUserId == users._AccountId)
sample_users = sample_posts_users.select(['Users.' + c for c in users.columns]).distinct()
sample_users.write.mode('overwrite')\
            .json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz',
                  compression='gzip')

#
# Get votes on those posts
#
votes = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Votes.df.parquet')\
             .alias('Votes')
sample_posts_votes = votes.join(F.broadcast(sample_qa_ids),
                                sample_qa_ids._SId == votes._PostId)
sample_votes = sample_posts_votes.drop('_SId')
sample_votes.write.mode('overwrite')\
            .json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz',
                  compression='gzip')


#
# Get the corresponding post histories
#
post_history = spark.read.parquet('s3://stackoverflow-events/06-24-2019/PostHistory.df.parquet')\
                    .alias('PostHistory')
sample_post_history = post_history.join(F.broadcast(sample_qa_ids),
                                        sample_qa_ids._SId == post_history._PostId)

sample_post_history.drop('_SId')\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/PostHistory.100K.Questions.jsonl.gz',
          compression='gzip')


#
# Get the corresponding badges
#
badges = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Badges.df.parquet')
user_ids = sample_users.select('_Id')\
                       .withColumn('_UId', sample_users._Id)\
                       .drop('_Id')\
                       .distinct()
sample_badges = badges.join(F.broadcast(user_ids),
                            badges._UserId == user_ids._UId).drop('_UId')
sample_badges.write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/Badges.100K.Questions.jsonl.gz',
          compression='gzip')

# Leave tags alone, tiny
# tags = spark.read.parquet('data/stackoverflow/parquet/Tags.df.parquet')
