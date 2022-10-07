# Big Data Final Project
# SSML - Spark Streaming for Machine Learning
# Spam Classifier

import time
import json
import pandas
import numpy as np
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, HashingTF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml.classification import NaiveBayes, LogisticRegression, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pyspark.sql.types import StructType, StructField, StringType
from uniplot import plot

sc = SparkContext(appName="BigDataProj")
ssc = StreamingContext(sc, 1)
lines = ssc.socketTextStream('localhost', 6100)
counts = lines.flatMap(lambda line: line.split("\n"))
lines.foreachRDD(lambda rdd: RDDtoDF(rdd))

def models():
    global nb, lr, svc
    nb = NaiveBayes(smoothing=1.0, modelType="gaussian")
    lr = LogisticRegression()
    svc = LinearSVC(maxIter = 10, regParam=0.1)


def modelHelper(df):
    models()
    df = df.withColumnRenamed('feature0','Subject').withColumnRenamed('feature1','Body').withColumnRenamed('feature2', 'Class')
    df = df.withColumn('lengthSubject',length(df['Subject']))
    df = df.withColumn('lengthBody',length(df['Body']))

    # encodes the Class column of to a column of label indices (0 for spam and 1 and ham)
    hamSpamToNum = StringIndexer(inputCol='Class',outputCol='Label')
    # converts the Body feature to lowercase and splits it by white spaces into a column Label
    tokenizer = Tokenizer(inputCol='Body', outputCol='tokenBody')
    # removes stop words
    stopRemove = StopWordsRemover(inputCol='tokenBody',outputCol='stopTokens')
    # converts text documents to vectors which give info about token counts
    countVector = CountVectorizer(inputCol='stopTokens',outputCol='cVector')
    # calculates inverse document frequency of cVector
    idf = IDF(inputCol='cVector', outputCol='tfIDF')
    # merges tfIDF and lengthBody into one vector column features
    cleanup = VectorAssembler(inputCols=['tfIDF','lengthBody'],outputCol='features')
    featureVectors = [hamSpamToNum,tokenizer,stopRemove,countVector,idf,cleanup]
    dataPipe = Pipeline(stages = featureVectors)

    # fits the pipeline to the original data
    cleaner = dataPipe.fit(df)
    cleanData = cleaner.transform(df)
    cleanData = cleanData.select(['label','features'])
    training,testing = cleanData.randomSplit([0.7,0.3])

    NBModel(training, testing)
    LRModel(training, testing)
    SVCModel(training, testing)

    print("----------------------") 

def NBModel(training, testing):
    spamPred = nb.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    testResults.show(n=10, truncate=True)
    print("Accuracy of Naive Bayes: {}%".format(acc*100))
    return acc

def LRModel(training, testing):
    spamPred = lr.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    testResults.show(n=10, truncate=True)
    print("Accuracy of Logistic Regression: {}%".format(acc*100))
    return acc

def SVCModel(training, testing):
    spamPred = svc.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    testResults.show(n=10, truncate=True)
    print("Accuracy of Support Vector Machine: {}%".format(acc*100))
    return acc

def RDDtoDF(rdd):
    spark = SparkSession(rdd.context)
    df_schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True), StructField("feature2", StringType(), True)])
    if not rdd.isEmpty():
        rdd = json.loads(rdd.collect()[0])
        pd_df = pandas.DataFrame.from_dict(rdd, orient = "index")
        rdd = spark.createDataFrame(pd_df, schema = df_schema)
        modelHelper(rdd)

ssc.start()
ssc.awaitTermination()
ssc.stop()
