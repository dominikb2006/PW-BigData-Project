# pyspark_job.py

import numpy as np
import re
from nltk import PorterStemmer

from pyspark import keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql.functions import to_timestamp, from_unixtime, when, col, regexp_replace, lower, udf, concat, lit, first
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, StringType


class Stemmer(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(Stemmer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)
        
    def _transform(self, df: DataFrame) -> DataFrame:
        
        def stem_words(words):
            return [PorterStemmer().stem(w) for w in words]
        
        t = ArrayType(StringType())
        out_col = self.getOutputCol()
        in_col = df[self.getInputCol()]
        return df.withColumn(out_col, udf(f=stem_words, returnType=t)(in_col))
    
    
def create_spark_session(is_local: bool):
    """
    Create spark session.
    Input: 
        is_local (bool) - True if run in local machine, False if in AWS EMR
    Output: 
        spark (SparkSession) - spark session connected to AWS EMR cluster
    """
    
    if is_local:
        spark = SparkSession \
        .builder \
        .appName("ML") \
        .config('spark.driver.host','127.0.0.1') \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "1g") \
        .config('spark.executor.cores', '2') \
        .getOrCreate()
        
    else:
        spark = SparkSession \
        .builder \
        .appName("ML") \
        .getOrCreate()
    
    return spark


def preprocessing(spark, input_path: str, is_test: bool):
    """
    Preprocess data
    Input:
        spark - SparkSession
        input_path (string) - path to data
        is_test (bool) - if True, limit data to 100 
    Output:
        df (DataFrame) - DataFrame from SparkSession
    """
    
    
    # read file
    df = spark.read.json(path=input_path)
    
    #change columns
    df = df.withColumn("label", df["overall"].cast(IntegerType()))
    # df = df.withColumn("asin", df["asin"].cast(DoubleType()))
    # df = df.withColumn("vote", when(col("vote") == "none", None).otherwise(col("vote")).cast("double"))
    # df = df.withColumn('style', df["style.Format:"]) 
    # df = df.withColumn('unixReviewTime', from_unixtime(df['unixReviewTime']))
    # df = df.withColumn("reviewTime", to_timestamp(df.reviewTime, 'MM d, yyyy'))

    df = df.drop('asin', 'image', 'reviewTime', 'reviewerID', 'reviewerName', 'reviewTime', 'unixReviewTime', 'overall'\
                , 'style', 'summary', 'verified', 'vote') 
    
    df = df.withColumn('reviewText', regexp_replace('reviewText', "\S+@\S+\s", ' '))\
                .withColumn('reviewText', regexp_replace('reviewText', "\n|\t", ' '))\
                .withColumn('reviewText', regexp_replace('reviewText', "\S*\d+\S*", ' '))\
                .withColumn('reviewText', regexp_replace('reviewText', "\s\W*\w\W*\s", ' '))\
                .withColumn('reviewText', regexp_replace('reviewText', "\W+", ' '))
    
    df = df.withColumn('reviewText', lower('reviewText'))
    # fill null values
    df = df.na.fill("null",["reviewText"]) 
    
    # limit data if test
    if is_test:
        df = df.limit(100)
        
    preprocessing_pipeline = Pipeline(
        stages=[
            RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W"),
            StopWordsRemover(inputCol='words', outputCol='words_cleaned'),
            Stemmer(inputCol='words_cleaned', outputCol='words_stemmed'),
            CountVectorizer(inputCol="words_stemmed", outputCol="term_freq", minDF=10.0), #, maxDF=0.5),
            IDF(inputCol="term_freq", outputCol="tfidf")
        ]
    )

    preprocessing_model = preprocessing_pipeline.fit(df)
    df = preprocessing_model.transform(df)
        
    return df
    

def process_book_data(spark, input_path, output_path, with_indexers: bool, is_test: bool, is_local: bool):
    """
    Process data - modeling, CrossValidation, Tuning Parameters
    Input:
        spark - SparkSession
        input_path (string) - path to data
        output_path (string) - path for output (best parameters)
        with_indexers (bool) - if True use additional Indexer: StringIndexer, VectorIndexer (it has no added value in this case)
        is_test (bool) - if True, limit data to 100 
        is_local (bool) - True if run in local machine, False if in AWS EMR    
    """
    
    df = preprocessing(spark, input_path, is_test)

    
    df = df.select(['tfidf', 'label'])
    
    print(df.cache())
    train_set, test_set = df.randomSplit([0.75, 0.25], seed=123)

    if with_indexers is True:
        """
        In our case this is not needed, it has no added value
        """
#         labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df)
#         featureIndexer = VectorIndexer(inputCol="tfidf", outputCol="indexedFeatures", maxCategories=4).fit(df)

#         r_forest = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
#         evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

#         param_grid = ParamGridBuilder() \
#         .addGrid(r_forest.maxDepth, [10, 20]) \
#         .addGrid(r_forest.maxBins, [5, 10]) \
#         .addGrid(r_forest.numTrees, [50, 100, 200]) \
#         .addGrid(r_forest.impurity, ['gini','entropy']) \
#         .build()
                
#         cv = CrossValidator(estimator=r_forest,
#                             estimatorParamMaps=param_grid,
#                             evaluator=evaluator,
#                             numFolds=5)

#         labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

#         pipeline = Pipeline(stages=[labelIndexer, 
#                                     featureIndexer, 
#                                     cv,
#                                     labelConverter
#                                    ])
        cv_stage = 2

    else:
        """
        Choose model
        """
        r_forest = RandomForestClassifier(labelCol="label", featuresCol="tfidf")

        """
        Choose metric
        """
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="accuracy")

        """
        Choose Hyperparameters for tuning
        """
        param_grid = ParamGridBuilder() \
        .addGrid(r_forest.bootstrap, [True]) \
        .addGrid(r_forest.maxDepth, [5, 15]) \
        .addGrid(r_forest.maxBins, [2, 6]) \
        .addGrid(r_forest.numTrees, [50, 100]) \
        .addGrid(r_forest.impurity, ['gini','entropy']) \
        .addGrid(r_forest.featureSubsetStrategy, ['auto']) \
        .addGrid(r_forest.minInstancesPerNode , [10, 25]) \
        .addGrid(r_forest.minInfoGain , [10, 25]) \
        .build()
        
        # .addGrid(r_forest.minWeightFractionPerNode, [0]) \
        
        """
        Definde CV and number of folds
        """
        cv = CrossValidator(estimator=r_forest,
                            estimatorParamMaps=param_grid,
                            evaluator=evaluator,
                            numFolds=3)


        pipeline = Pipeline(stages=[cv])
        cv_stage = 0


    model = pipeline.fit(train_set)
    predictions = model.transform(test_set)
    
    """
    Get best params from model and write it to file
    """
    hyperparams = model.stages[cv_stage].getEstimatorParamMaps()[np.argmax(model.stages[cv_stage].avgMetrics)]
    hyper_list = []

    for i in range(len(hyperparams.items())):
        hyper_name = re.search("name='(.+?)'", str([x for x in hyperparams.items()][i])).group(1)
        hyper_value = [x for x in hyperparams.items()][i][1]

        hyper_list.append({hyper_name: hyper_value})

    # Read into Spark DataFrame
    params = spark.read.json(spark.sparkContext.parallelize(hyper_list))
    # Eliminate null rows
    params = params.agg(*[first(x, ignorenulls=True) for x in params.columns])
    
    if not is_local:
        params.write.format("com.databricks.spark.csv").option("header", "true").save(
            path = output_path, mode = "overwrite")
        
    else:
        params.show()
    
    
def main():
    """
    Main function
    Define variables here
    WITH_INDEXERS (bool) - if True use additional Indexer: StringIndexer, VectorIndexer (it has no added value in this case)
    IS_TEST (bool) - if True, limit data to 100 
    IS_LOCAL (bool) - True if run in local machine, False if in AWS EMR    
    
    """
    
    WITH_INDEXERS = False
    IS_TEST = False
    IS_LOCAL = False
    
    if IS_LOCAL:
        INPUT_PATH = r'C:\Users\dominik.brys\OneDrive - Accenture\DB\PW-Big_Data\Projekt\Books_5\Books_5_limited_v2.json'
        OUTPUT_PATH = r'C:\Users\dominik.brys\OneDrive - Accenture\DB\PW-Big_Data\Projekt\output\model'
    else:
        bucket = 's3://emr-project-v1-db'
        input_file = '/Books_5.json'
        output_file = '/output_params'
        INPUT_PATH = bucket + input_file
        OUTPUT_PATH = bucket + output_file
    
    spark = create_spark_session(IS_LOCAL)
    process_book_data(spark, INPUT_PATH, OUTPUT_PATH, WITH_INDEXERS, IS_TEST, IS_LOCAL)

    
if __name__ == '__main__':
    main()