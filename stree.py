from pyspark import sql
from pyspark.ml.feature import StringIndexer
from functools import reduce
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = sql.SparkSession.builder.master("local").appName("Decision Trees").getOrCreate()

df = spark.read.csv("./data/mushrooms.data.txt", header=True, inferSchema=True)


def string_to_index(df, label):
    """用来增加列的"""
    return StringIndexer(inputCol=label, outputCol="i-" + label).fit(df).transform(df)


# 用 reduce 在每个 category 上操作
category = ["cap-shape", "cap-surface", "cap-color"]
df = reduce(string_to_index, category, df)

# 合并成一个向量列
df = VectorAssembler(inputCols=["i-" + x for x in category], outputCol="features").transform(df)

# 处理标签
df = StringIndexer(inputCol="edible?", outputCol="label").fit(df).transform(df)

# 训练决策树模型
tree = DecisionTreeClassifier()
model = tree.fit(df)

# 评估
bce = BinaryClassificationEvaluator()
result = bce.evaluate(model.transform(df))
print(result)
