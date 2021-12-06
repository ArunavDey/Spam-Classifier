# Big Data Final Project
# SSML - Spark Streaming for Machine Learning
# Spam Classifier
# BD2_066_167_232_339

import time
import json
import pandas
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, HashingTF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 

sc = SparkContext(appName="BigDataProj")
ssc = StreamingContext(sc, 1)
lines = ssc.socketTextStream('localhost', 6100)
counts = lines.flatMap(lambda line: line.split("\n"))
counts.foreachRDD(lambda rdd: RDDtoDF(rdd))

def model(df):
    df = df.withColumnRenamed('feature0','Subject').withColumnRenamed('feature1','Body').withColumnRenamed('feature2', 'Class')
    df = df.withColumn('lengthSubject',length(df['Subject']))
    df = df.withColumn('lengthBody',length(df['Body']))

    hamSpamToNum = StringIndexer(inputCol='Class',outputCol='Label')
    tokenizer = Tokenizer(inputCol="Body", outputCol="tokenBody")
    stopRemove = StopWordsRemover(inputCol='tokenBody',outputCol='stopTokens')
    countVector = CountVectorizer(inputCol='stopTokens',outputCol='cVector')
    idf = IDF(inputCol="cVector", outputCol="tfIDF")
    cleanup = VectorAssembler(inputCols=['tfIDF','lengthBody'],outputCol='features')
    featureVectors = [hamSpamToNum,tokenizer,stopRemove,countVector,idf,cleanup]
    dataPipe = Pipeline(stages = featureVectors)

    cleaner = dataPipe.fit(df)
    cleanData = cleaner.transform(df)
    cleanData = cleanData.select(['label','features'])
    training,testing = cleanData.randomSplit([0.7,0.3])
    model2(training, testing)

def model2(training, testing):
    nb = NaiveBayes()
    spamPredNB = nb.fit(training)
    testResults = spamPredNB.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    testResults.show()
    print("Accuracy of Naive Bayes at predicting spam was: {}%".format(acc*100))

def RDDtoDF(rdd):
    #TODO
    spark = SparkSession(rdd.context)
    df_schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])
    if not rdd.isEmpty():
        rdd = json.loads(rdd.collect()[0])
        pd_df = pandas.DataFrame.from_dict(rdd, orient = "index")
        rdd = spark.createDataFrame(pd_df, schema = df_schema)
        model(rdd)


ssc.start()
ssc.awaitTermination()
ssc.stop()
