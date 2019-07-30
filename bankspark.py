import findspark
findspark.init()


import pyspark # only run after findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName('bankspark2').getOrCreate()

df = spark.read.format("csv").option("header", "true").option("delimiter", ";").load("bank.csv")
df.show(4)

cols = ['age', 'job', 'marital','education', 'default', 'balance', 'housing', 'loan', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df2 = df.select(cols)
df2.show(4)

df2.createOrReplaceTempView("bank")
df2 = spark.sql("SELECT * FROM bank WHERE bank.poutcome='success' or bank.poutcome='failure'")
df2.createOrReplaceTempView("bank")
df2 =  df2.filter(~df.age.contains("unknown"))
df2 =  df2.filter(~df.job.contains("unknown"))
df2 =  df2.filter(~df.marital.contains("unknown"))
df2 =  df2.filter(~df.education.contains("unknown"))
df2 =  df2.filter(~df.default.contains("unknown"))
df2 =  df2.filter(~df.balance.contains("unknown"))
df2 =  df2.filter(~df.housing.contains("unknown"))
df2 =  df2.filter(~df.loan.contains("unknown"))
df2 =  df2.filter(~df.campaign.contains("unknown"))
df2 =  df2.filter(~df.pdays.contains("unknown"))
df2 =  df2.filter(~df.previous.contains("unknown"))
df2 =  df2.filter(~df.poutcome.contains("unknown"))
df2 =  df2.filter(~df.y.contains("unknown"))
df2.show(4)

stringIndexer = StringIndexer(inputCol="job", outputCol="job_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="job_index", outputCol="job_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="marital", outputCol="marital_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="marital_index", outputCol="marital_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="education", outputCol="education_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="education_index", outputCol="education_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="default", outputCol="default_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="default_index", outputCol="default_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="housing", outputCol="housing_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="housing_index", outputCol="housing_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="loan", outputCol="loan_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="loan_index", outputCol="loan_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="poutcome", outputCol="poutcome_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="poutcome_index", outputCol="poutcome_vec")
encoded = encoder.transform(indexed)
df2 = encoded

stringIndexer = StringIndexer(inputCol="y", outputCol="y_index")
model = stringIndexer.fit(df2)
indexed = model.transform(df2)
encoder = OneHotEncoder(dropLast=False, inputCol="y_index", outputCol="y_vec")
encoded = encoder.transform(indexed)
df2 = encoded

df2.show(4)


cols = ['job', 'marital','education', 'default', 'housing', 'loan', 'poutcome', 'y']
for col in cols:
    scaler = MinMaxScaler(inputCol=col+"_vec", outputCol=col+"_vec_scaled")
    scalerModel = scaler.fit(df2)
    scaledData = scalerModel.transform(df2)
    df2 = scaledData
df2.show(4)

vecAssembler = VectorAssembler(inputCols=[ 'job_vec_scaled', 'marital_vec_scaled','education_vec_scaled', 'default_vec_scaled', 'housing_vec_scaled', 'loan_vec_scaled', 'poutcome_vec_scaled'], outputCol='features')
df3 = vecAssembler.transform(df2)
df3.show(4)

kmeans = KMeans(k=2, seed=1)
model = kmeans.fit(df3.select("features"))
transformed = model.transform(df3)
transformed.show(4)

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df3)

result = model.transform(df3).select("pcaFeatures")
pandasDf = result.toPandas()
dataX = []
dataY = []
for vec in pandasDf.values:
    dataX.extend([vec[0][0]])
    dataY.extend([vec[0][1]])
plt.scatter(dataX, dataY)
plt.show()

sns.scatterplot(dataX, dataY)
plt.title('PCA Features')
plt.show()