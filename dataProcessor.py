from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import StringType
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from functools import reduce
from pyspark.sql import DataFrame
import math

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


conf = SparkConf().setAppName("dataProcessor")
sc = SparkContext(conf = conf)
sqc = SQLContext(sc)

"""
with open('carriers.csv') as f:
    carriers = f.readlines()

carriers = [x.strip() for x in carriers]

with open('airports.csv') as f:
    airport = f.readlines()

airport = [x.strip() for x in airport]

"""

df1 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2003.csv')

df2 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2004.csv')

df3 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2005.csv')

df4 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2006.csv')

df5 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2007.csv')
"""
df6 = sqc.read.format("com.databricks.spark.csv") \
    .options(header = 'true', inferschema = 'true') \
    .load('2008.csv')
"""

data = unionAll(df1, df2, df3, df4, df5)
data2 = data.select('ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Distance', 'Cancelled', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'WeatherDelay').replace('NA', '0')
"""
name = 'UniqueCarrier'
udf = UserDefinedFunction(lambda x: carriers.index(x) if x in carriers else 0, StringType())
data3 = data2.select(*[udf(column).alias(name) if column == name else column for column in data2.columns])


name = 'Origin'
udf = UserDefinedFunction(lambda x: airport.index(x) if x in airport else 0, StringType())
data4 = data3.select(*[udf(column).alias(name) if column == name else column for column in data3.columns])

name = 'Dest'
udf = UserDefinedFunction(lambda x: airport.index(x) if x in airport else 0, StringType())
data5 = data4.select(*[udf(column).alias(name) if column == name else column for column in data4.columns])

"""
#test set
#data7 = df6.select('ArrTime', 'CRSArrTime', 'FlightNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Distance', 'Cancelled', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'WeatherDelay').replace('NA', '0')

#training
data6 = data2.map( lambda line: LabeledPoint( line[13] , [line[0:13]] ) )
#trainingData = data2.map( lambda line: LabeledPoint( line[13] , [line[0:13]] ) )
#testData = data7.map( lambda line: LabeledPoint( line[13] , [line[0:13]] ) )
(trainingData, testData) = data6.randomSplit([0.7, 0.3])
model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo={}, impurity='variance', maxDepth=8, maxBins=256)
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMAE = labelsAndPredictions.map(lambda (v, p): math.fabs(v-p)).sum() / float(testData.count())
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Learned regression tree model:')
print(model.toDebugString())
print('Test Root Mean Squared Error = ' + str(math.sqrt(testMSE)))
print('Test MAE = ' + str(testMAE))
