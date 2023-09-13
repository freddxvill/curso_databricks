# Databricks notebook source
# MAGIC %md
# MAGIC ## Creacion de tabla para regresión

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField
 
schema = StructType([
  StructField("longitude", DoubleType(), True),
  StructField("latitude", DoubleType(), True),
  StructField("housing_median_age", DoubleType(), True),
  StructField("total_rooms", DoubleType(), True),
  StructField("total_bedrooms", DoubleType(), True),
  StructField("population", DoubleType(), True),
  StructField("households", DoubleType(), True),
  StructField("median_income", DoubleType(), True),
  StructField("median_house_value", DoubleType(), True),
  StructField("ocean_proximity", StringType(), True)
])
# sesion content y sesion por defecto
housing_df = spark.read.format("csv").schema(schema).option("header", "true").load("/FileStore/curso/housing.csv")

# COMMAND ----------

# conversion a delta table en databricks
housing_df.write.saveAsTable("default.housing_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from housing_t

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creacion de tabla para clasificación

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField
 
schema = StructType([
  StructField("age", DoubleType(), True),
  StructField("workclass", StringType(), True),
  StructField("fnlwgt", DoubleType(), True),
  StructField("education", StringType(), True),
  StructField("education_num", DoubleType(), True),
  StructField("marital_status", StringType(), True),
  StructField("occupation", StringType(), True),
  StructField("relationship", StringType(), True),
  StructField("race", StringType(), True),
  StructField("sex", StringType(), True),
  StructField("capital_gain", DoubleType(), True),
  StructField("capital_loss", DoubleType(), True),
  StructField("hours_per_week", DoubleType(), True),
  StructField("native_country", StringType(), True),
  StructField("income", StringType(), True)
])
census_df = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")

# COMMAND ----------

census_df.write.saveAsTable("default.census_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM census_t

# COMMAND ----------


