
<a href="http://www.calstatela.edu/centers/hipic"><img align="left" src="https://avatars2.githubusercontent.com/u/4156894?v=3&s=100"><image/>
</a>
<img align="right" alt="California State University, Los Angeles" src="http://www.calstatela.edu/sites/default/files/groups/California%20State%20University%2C%20Los%20Angeles/master_logo_full_color_horizontal_centered.svg" style="width: 360px;"/>

#    CIS5560 Term Project Tutorial


------
#### Authors: Bhagyashree Bhagwat, Niklas Meher, Priyanka Purushu

#### Instructor: [Jongwook Woo](https://www.linkedin.com/in/jongwook-woo-7081a85)

#### Date: 05/18/2017


# Fraud Transaction Detection using Spark ML

## Abstract:
This project aims at analyzing data and providing insights on Financial Fraud Detection using Spark ML. The dataset contains a sample of real transactions extracted from one month of financial logs from a mobile money service.The original logs were provided by the multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world. This dataset is scaled down to 1/4 of the original dataset and it is created just for Kaggle. 

## Dataset specification
- File format : *.csv* format
- url : *(https://www.kaggle.com/ntnu-testimon/paysim1)*

We took the whole dataset and tried three different classification models in spark Ml and performed data preparation, model building and validation and finally model evaluation and interpretation.

## Import Spark SQL and Spark ML Libraries
Import all the Spark SQL and ML libraries as mentioned below. This is neccessary to access the functions available in those libraries.


```python
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
```

## Load Source Data

In order to upload the .csv file i.e. our dataset follow the below mentioned steps:-

1. Go to tables and click on create table. In the data source choose *file* if it is not in the *DBFS (Databricks File System)* and import the .csv file from the local system.
2. The .csv file will be stored in an unique location in your DBFS and the path where the file is stored will appear against *uploaded to DBFS*. Note down the file path as it is mandatory to provide this path when we call spark.read function.
3. Click on the *peview table* option and provide a table name, choose the file type e.g. CSV and the choose the delimiter based on your file.
4. Once the file is uploaded to DBFS now call the function *spark.read.format().option().load()* and provide the file path is the load function. Assign this read function to a variable as mentioned in the code below.


```python
# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_f63cbb38899d47179c49ed4a7cf03ccf(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '265dd6d8a99a4549a24ac9574846808d')
    
    
    hconf.set(prefix + '.username', '886a93bbc2564a539f02a62ed61e1a61')
    hconf.set(prefix + '.password', 'h8bjj]J[1DME7LnC')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_f63cbb38899d47179c49ed4a7cf03ccf(name)

spark = SparkSession.builder.getOrCreate()

# change the file path in load with your own file path
df = spark.read\
  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
  .option('header', 'true')\
  .option("inferSchema", "true")\
  .load('swift://ITBSProjectFraudDetection.' + name + '/frauddetectionsmall.csv')
df.take(5)
df.dtypes

```

## Visualizations

Visualization of the data to get a better understanding of it. In this project we have done bar plot and scatter plot visualisation of the data.

**1. Insights from Bar plot**

   - Amount of transaction grouped by type
    - We see that most of the fraud transactions are made with *CASH_OUT* and *PAYMENTS*
   - Fraud transaction grouped by *CASH_OUT* and *PAYMENT*
    - Fraud transactions are only made with the type *CASH_OUT* and *TRANSFER*. This is an interesting fact since the type TRANSFER is in the fourth place when it comes to the number of transactions. 
    
**2. Insights from Scatter plot**

 - Since the points on scatter plots are alligned closely in a linear manner we can determine that there is a direct relatioship between the attributes 'newbalanceDest' and 'oldbalanceDest' as well as between 'newbalanceOrig' and 'oldbalanceOrg'.


```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

fraud = df.toPandas()
f, ax = plt.subplots(1, 1, figsize=(8, 8))
fraud.type.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
plt.show()
plt.figure(0)
cond = (fraud['isFraud'] >= 1)
taf = fraud[cond].type.value_counts().plot(kind='bar',  title="Fraud transactions grouped by type")
plt.show(taf)
plt.figure(1)
cond2 = (fraud['isFraud'] < 1)
taf2 = fraud[cond2].type.value_counts().plot(kind='bar',  title="No fraud transactions grouped by type")
plt.show(taf2)
fraud.hist(column='isFlaggedFraud', bins=5)
plt.show()
plt.figure(2)
medianprops = dict(linestyle='-', linewidth=2, color='blue')
bx1 = fraud[cond2].boxplot(column=['oldbalanceDest', 'newbalanceDest'], by='isFraud', medianprops=medianprops)
bx2 = fraud[cond2].boxplot(column=['oldbalanceOrg', 'newbalanceOrig'], by='isFraud', medianprops=medianprops)
bx3 = fraud[cond2].boxplot(column=['amount'], by='isFraud', medianprops=medianprops)
#bx4 = fraud[cond].boxplot(column=['type'], by='isFraud', medianprops=medianprops)
plt.show(bx1)
plt.show(bx2)
plt.show(bx3)
plt.figure(3)

```


```python
fraud.head(5)
```

## Scatter Plots

Next, we want to find correlations between attributes using scatter plots. It's visible that there are correlations between the attributes 'newbalanceDest' and 'oldbalanceDest' as well as between 'newbalanceOrig' and 'oldbalanceOrg'.


```python
plt.figure(4)
sc1 = fraud.plot.scatter(x='oldbalanceDest', y='newbalanceDest')
sc2 = fraud.plot.scatter(x='oldbalanceOrg', y='newbalanceOrig')
sc1 = fraud.plot.scatter(x='oldbalanceDest', y='oldbalanceOrg')
sc2 = fraud.plot.scatter(x='oldbalanceOrg', y='newbalanceDest')
sc3 = fraud.plot.scatter(x='amount', y='isFraud')
sc4 = fraud.plot.scatter(x='oldbalanceDest', y='isFraud')
sc5 = fraud.plot.scatter(x='newbalanceDest', y='isFraud')
sc6 = fraud.plot.scatter(x='oldbalanceOrg', y='isFraud')
sc7 = fraud.plot.scatter(x='newbalanceOrig', y='isFraud')
plt.show(sc1)
plt.show(sc2)
plt.show(sc3)
plt.show(sc4)
plt.show(sc5)
plt.show(sc6)
plt.show(sc7)
```

## Clear the data

In the next step, we will select the columns that are useful for our model. This is done in order to avoid the ppresence of outliners in the model. 

1. The columns that we have removed are:
  -   step
  -  nameDest
  -  nameOrig
  -  IsFlaggedFraud

2. Columns retained are:
  -  amount
  -  oldbalanceOrg
  -  newbalanceOrig
  -  oldbalanceDest
  -  newbalanceDest


```python
df2 = df.select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", (col("isFraud").cast("Int").alias("label")))
df2.take(5)
```

## Split the data
In the next step we split the data in a train and test set. We have split the data in the ratio of **70 to 30**.


```python
splits = df2.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print "Training Rows:", train_rows, " Testing Rows:", test_rows
train.show(5)
test.show(5)
```

## Define a pipeline
Pipeline provides a simple construction, tuning and testing for ML workflows.

## Algorithms used to Train the model

In this project we have used the following algorithm to train our model:

1. Random Forest Classifier (RF)
2. Decision Tree Classifier (DT) 
3. Logistic Regression (LR).


```python
strIdx = StringIndexer(inputCol = "type", outputCol = "typeCat")
labelIdx = StringIndexer(inputCol = "label", outputCol = "idxLabel")
# number is meaningful so that it should be number features
catVect = VectorAssembler(inputCols = ["typeCat"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")
numVect = VectorAssembler(inputCols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")
featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")

cl = []
pipeline = []


cl.insert(0, DecisionTreeClassifier(labelCol="idxLabel", featuresCol="features"))
cl.insert(1, RandomForestClassifier(labelCol="idxLabel", featuresCol="features"))
cl.insert(2, LogisticRegression(labelCol="idxLabel", featuresCol="features"))


# Pipeline process the series of transformation above, which is 7 transformation
for i in range(3):
    pipeline.insert(i, Pipeline(stages=[strIdx, labelIdx, catVect, catIdx, numVect, minMax, featVect, cl[i]]))
    #piplineModel = pipeline.fit(train)
print "Pipeline complete!"
```

## Train Validation Split

we used a train validation split instead of cross validation for every model because it takes much less time to train the model with the train validation split.


```python
model = []

#When using the whole dataset, please use the TrainValidationSplit instead of the CrossValidator!

paramGrid = (ParamGridBuilder().addGrid(cl[0].impurity, ("gini", "entropy")).addGrid(cl[0].maxDepth, [5, 10, 20]).addGrid(cl[0].maxBins, [5, 10, 20]).build())
cv = CrossValidator(estimator=pipeline[0], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, numFolds=5)
#cv = TrainValidationSplit(estimator=pipeline[0], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(0, cv.fit(train))
print "Model 1 completed"


paramGrid2 = (ParamGridBuilder().addGrid(cl[1].impurity, ("gini", "entropy")).addGrid(cl[1].maxDepth, [5, 10, 20]).addGrid(cl[1].maxBins, [5, 10, 20]).build())
cv2 = CrossValidator(estimator=pipeline[1], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid2, numFolds=5)
#cv2 = TrainValidationSplit(estimator=pipeline[1], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)
model.insert(1, cv2.fit(train))
print "Model 2 completed"

paramGrid3 = (ParamGridBuilder().addGrid(cl[2].regParam, [0.01, 0.5, 2.0]).addGrid(cl[2].threshold, [0.30, 0.35, 0.5]).addGrid(cl[2].maxIter, [1, 5]).addGrid(cl[2].elasticNetParam, [0.0, 0.5, 1]).build())
cv3 = CrossValidator(estimator=pipeline[2], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid3, numFolds=5)
#cv3 = TrainValidationSplit(estimator=pipeline[2], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model.insert(2, cv3.fit(train))
print "Model 3 completed"
```

## Test the model
We transform the test dataframe to generate label predictions.


```python
'''predictions = model.transform(test)
predicted = predictions.select("features", "prediction", "probability", "trueLabel")
predicted.show(100, truncate=False)
for row in predicted.collect():
    print row'''
prediction = [] 
predicted = []
for i in range(3):
  prediction.insert(i, model[i].transform(test))
  predicted.insert(i, prediction[i].select("features", "prediction", "probability", "trueLabel"))
  predicted[i].show(30)
```

## Evaluation

In this step we evaluate the model. The important measuring metrics that we have based our evaluation on are **Recall, Precision and AUC**. The values of these metrics helps in indicating how good we are in detecting fraud transactions when the transaction is actually a fraud.

Recall and Precision are calculated based on the TP,FP and FN values from the confusion matrix. The formulaes to calculate Recall and precision are given below.
1. **Recall    = TP / (TP + FN)**
2. **Precision = TP / (TP + FP)**

Description of the Terms used above are:
-  TP : True positives 
-  FP : False Positives i.e. values that are not positive
-  FN : False Negative i.e. values that are positive


```python
#from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="trueLabel", rawPredictionCol="prediction")
for i in range(3):
    #evaluator = MulticlassClassificationEvaluator(
    #labelCol="trueLabel", predictionCol="prediction", metricName="weightedRecall")
    areUPR = evaluator.evaluate(predicted[i], {evaluator.metricName: "areaUnderPR"})
    areUROC = evaluator.evaluate(predicted[i], {evaluator.metricName: "areaUnderROC"})
    print("AreaUnderPR = %g " % (areUPR))
    
    print("AreaUnderROC = %g " % (areUROC))

    tp = float(predicted[i].filter("prediction == 1.0 AND truelabel == 1").count())
    fp = float(predicted[i].filter("prediction == 1.0 AND truelabel == 0").count())
    tn = float(predicted[i].filter("prediction == 0.0 AND truelabel == 0").count())
    fn = float(predicted[i].filter("prediction == 0.0 AND truelabel == 1").count())

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision = %g " % (precision))
    print("Recall = %g " % (recall))

    metrics = sqlContext.createDataFrame([
    ("TP", tp),
    ("FP", fp),
    ("TN", tn),
    ("FN", fn),
    ("Precision", tp / (tp + fp)),
    ("Recall", tp / (tp + fn))],["metric", "value"])
    metrics.show()

```


```python
%%html
<style>
table {float:left}
</style>
```


<style>
table {float:left}
</style>


## Conclusion

The metrics values for the 3 algorithms are as follows:

| **Model**                   | **Area Under ROC**| **Precision**| **Recall **| 
|:----------------------- |:-------------:| :--------:|:-------:|
| DecisionTreeClassifier  | 0.839343      | 0.965338 |0.678716|
| RandomForestClassifier  | 0.85963       | 0.92674  |0.719334|
| LogisticRegression      | 0.726389      | 0.845979 |0.452884|


The RandomForestClassifier has the best recall score as compared to DecisionTreeClassifier and LogisticRegression.


```python

```
