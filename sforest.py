from pyspark.sql import SparkSession
from functools import reduce
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def string_to_index(df, label):
    """用来增加列的"""
    return StringIndexer(inputCol=label, outputCol="i-" + label).fit(df).transform(df)


spark = SparkSession.builder.master("local").appName("Decision Trees").getOrCreate()
df = spark.read.csv("./data/mushrooms.data.txt", header=True, inferSchema=True)

category = df.columns
category.pop(category.index("edible?"))

df = reduce(string_to_index, category, df)
index = ["i-" + x for x in category]

df = VectorAssembler(inputCols=index, outputCol="features").transform(df)
df = StringIndexer(inputCol="edible?", outputCol="label").fit(df).transform(df)

# 训练
forest = RandomForestClassifier()
grid = ParamGridBuilder().addGrid(forest.maxDepth, [0, 2]).build()
bce = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=forest, estimatorParamMaps=grid, evaluator=bce, parallelism=4)

cv_model = cv.fit(df)

area_under_curve = bce.evaluate(cv_model.transform(df))
print("Random Forest AUC: {:0.4f}".format(area_under_curve))

print(cv_model.bestModel.toDebugString)
