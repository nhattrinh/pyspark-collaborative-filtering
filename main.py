import sys
import re
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

conf = SparkConf()
sc = SparkContext(conf=conf)

conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", "8G")
conf.set("spark.executor.cores", "4")

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

# Load and parse the data
data = sc.textFile('./train.dat')
ratings = data.map(lambda line: line.split('\t'))\
    .map(lambda line: Rating(int(line[0]), int(line[1]), float(line[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

# Evaluate the model on training data
testdata = sc.textFile('./test.dat')
testratings = testdata.map(lambda line: line.split('\t'))\

predictions = model.predictAll(testratings.map(lambda d: (int(d[0]), int(d[1]))))

# Write the RDD to format.dat file, where each row is a score prediction
predictions.map(lambda rating: int(rating[3])) \
    .saveAsTextFile('./format.dat')

sc.stop()