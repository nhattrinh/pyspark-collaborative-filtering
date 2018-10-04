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
    .map(lambda rating: Rating(int(rating[0]), int(rating[1]), float(rating[2])))

# Build the recommendation model using Alternating Least Squares
rank = 10
num_iterations = 10
model = ALS.train(ratings, rank, num_iterations)

# Evaluate the model on training data
test_data = sc.textFile('./test.dat')
test_ratings = test_data.map(lambda line: line.split('\t'))

predictions = model.predictAll(test_ratings.map(lambda d: (int(d[0]), int(d[1]))))

# Write the RDD to format.dat file, where each row is a score prediction
format_file = open('./format.dat','w')

for prediction in predictions.collect():
    if (int(prediction[2]) < 0):
        format_file.write('{}\n'.format(int(0)))
    format_file.write('{}\n'.format(prediction[2]))

format_file.close()

sc.stop()