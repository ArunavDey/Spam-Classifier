# Big Data Final Project
# SSML - Spark Streaming for Machine Learning
# Spam Classifier
# BD2_066_167_232_339

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

sc = SparkContext(appName="BigDataProj")
ssc = StreamingContext(sc, 1)
lines = ssc.socketTextStream('localhost', 6100)
counts = lines.flatMap(lambda line: line.split("\n"))
counts.foreachRDD(lambda rdd: RDDtoDF(rdd))

def modelHelper(df):
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
    # training.select(['features']).show()
    # training.select(['label']).show()
    # print(type(training))
    print("----------------------") 
    NBModel(training, testing)
    LRModel(training, testing)
    SVMModel(training, testing)
    print("----------------------") 
    # skNB(training.select(['features']), training.select(['label']), testing.select(['features']))

def skNB(training, target, testing):
    nb = GaussianNB()

    train = np.array(training.collect())
    test = np.array(testing.collect())
    tar = np.array(target.collect())

    nsamples, nx, ny = train.shape
    train_dataset = train.reshape((nsamples,nx*ny))


    msamples, mx, my = tar.shape
    target_dataset = tar.reshape((msamples,mx*my))

    osamples, ox, oy = test.shape
    test_dataset = test.reshape((osamples,ox*oy))

    nb.partial_fit(train_dataset, target_dataset, ['0.0', '1.0'])
    spamPred = np.array(nb.predict(test_dataset))
    acc = accuracy_score(test_dataset, spamPred)
    print(acc)


def NBModel(training, testing):
    nb = NaiveBayes(smoothing=1.0, modelType="gaussian")
    spamPred = nb.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    # testResults.show(n=10, truncate=True)
    print("Accuracy of Naive Bayes: {}%".format(acc*100))

def LRModel(training, testing):
    lr = LogisticRegression()
    spamPred = lr.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    # testResults.show(n=10, truncate=True)
    print("Accuracy of Logistic Regression: {}%".format(acc*100))

def SVMModel(training, testing):
    svm = LinearSVC(maxIter = 10, regParam=0.1)
    spamPred = svm.fit(training)
    testResults = spamPred.transform(testing)
    acc_eval = MulticlassClassificationEvaluator()
    acc = acc_eval.evaluate(testResults)
    # testResults.show(n=10, truncate=True)
    print("Accuracy of Support Vector Machine: {}%".format(acc*100))

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
