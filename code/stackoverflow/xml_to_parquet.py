from pyspark.sql import SparkSession


# spark = SparkSession.builder.appName('Deep Products').getOrCreate()
# sc = spark.sparkContext


# Spark-XML DataFrame method
votes_df = spark.read.format('xml').options(rowTag='row').options(rootTag='votes').load('s3://stackoverflow-events/06-24-2019/Votes.xml.lzo')
votes_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Votes.df.parquet')

posts_df = spark.read.format('xml').options(rowTag='row').options(rootTag='posts').load('s3://stackoverflow-events/06-24-2019/Posts.xml.lzo')
posts_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Posts.df.parquet')

users_df = spark.read.format('xml').options(rowTag='row').options(rootTag='users').load('s3://stackoverflow-events/06-24-2019/Users.xml.lzo')
users_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Users.df.parquet')

tags_df = spark.read.format('xml').options(rowTag='row').options(rootTag='tags').load('s3://stackoverflow-events/06-24-2019/Tags.xml.lzo')
tags_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Tags.df.parquet')

badges_df = spark.read.format('xml').options(rowTag='row').options(rootTag='badges').load('s3://stackoverflow-events/06-24-2019/Badges.xml.lzo')
badges_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Badges.df.parquet')

comments_df = spark.read.format('xml').options(rowTag='row').options(rootTag='comments').load('s3://stackoverflow-events/06-24-2019/Comments.xml.lzo')
comments_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/Comments.df.parquet')

history_df = spark.read.format('xml').options(rowTag='row').options(rootTag='posthistory').load('s3://stackoverflow-events/06-24-2019/PostHistory.xml.lzo')
history_df.write.mode('overwrite').parquet('s3://stackoverflow-events/06-24-2019/PostHistory.df.parquet')
