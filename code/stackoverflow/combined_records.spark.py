

questions = questions.limit(1000000)
questions\
    .write.mode('overwrite')\
    .parquet('s3://stackoverflow-events/07-19-2019/Questions.Answered.1M.parquet')
sample_posts = spark.read.parquet('s3://stackoverflow-events/07-19-2019/Questions.Answered.1M.parquet')


# Write the Q&A post IDs for later joins
sample_qa_ids = sample_posts.select('_Id').withColumnRenamed('_Id', '_SId')
sample_qa_ids\
    .coalesce(1)\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/SampleIds.100K.Questions.jsonl.gz',
          compression='gzip')
sample_qa_ids = spark.read.json('s3://stackoverflow-events/06-24-2019/SampleIds.100K.Questions.jsonl.gz')


#
# Get the corresponding comments
#
comments = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Comments.df.parquet')
sample_comments = comments.join(sample_qa_ids,
                                sample_qa_ids._SId == comments._PostId)
print('Total sample comments: {:,}'.format(sample_comments.count()))
sample_comments.drop('_SId')\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz',
          compression='gzip')
sample_comments = spark.read.json('s3://stackoverflow-events/06-24-2019/Comments.100K.Questions.jsonl.gz')

#
# Get users participating in those posts
#
users = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Users.df.parquet')\
             .alias('Users')
sample_posts_users = sample_posts.join(users,
                                       sample_posts._OwnerUserId == users._Id)
print('Full users sample: {:,}'.format(sample_posts_users.count()))
sample_users = sample_posts_users.select(['Users.' + c for c in users.columns]).distinct()
print('Distinct users sample: {:,}'.format(sample_users.count()))
sample_users.write.mode('overwrite')\
            .json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz',
                  compression='gzip')
sample_users = spark.read.json('s3://stackoverflow-events/06-24-2019/Users.100K.Questions.jsonl.gz')

#
# Get votes on those posts
#
votes = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Votes.df.parquet')\
             .alias('Votes')
sample_posts_votes = votes.join(sample_qa_ids,
                                sample_qa_ids._SId == votes._PostId)
print('Full votes sample: {:,}'.format(sample_posts_votes.count()))
sample_votes = sample_posts_votes.drop('_SId')
sample_votes.write.mode('overwrite')\
            .json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz',
                  compression='gzip')
sample_votes = spark.read.json('s3://stackoverflow-events/06-24-2019/Votes.100K.Questions.jsonl.gz')

#
# Get the corresponding post histories
#
post_history = spark.read.parquet('s3://stackoverflow-events/06-24-2019/PostHistory.df.parquet')\
                    .alias('PostHistory')
sample_post_history = post_history.join(sample_qa_ids,
                                        sample_qa_ids._SId == post_history._PostId)
print('Full post history sample: {:,}'.format(sample_post_history.count()))
sample_post_history.drop('_SId')\
    .write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/PostHistory.100K.Questions.jsonl.gz',
          compression='gzip')
sample_post_history = spark.read.json('s3://stackoverflow-events/06-24-2019/PostHistory.100K.Questions.jsonl.gz')

#
# Get the corresponding badges
#
badges = spark.read.parquet('s3://stackoverflow-events/06-24-2019/Badges.df.parquet')
user_ids = sample_users.select('_Id')\
                       .withColumn('_UId', sample_users._Id)\
                       .drop('_Id')\
                       .distinct()
sample_badges = badges.join(user_ids,
                            badges._UserId == user_ids._UId).drop('_UId')
print('Full badges sample: {:,}'.format(sample_badges.count()))
sample_badges.write.mode('overwrite')\
    .json('s3://stackoverflow-events/06-24-2019/Badges.100K.Questions.jsonl.gz',
          compression='gzip')
sample_badges = spark.read.json('s3://stackoverflow-events/06-24-2019/Badges.100K.Questions.jsonl.gz')

# Leave tags alone, tiny
# tags = spark.read.parquet('data/stackoverflow/parquet/Tags.df.parquet')
