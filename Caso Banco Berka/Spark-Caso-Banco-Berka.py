# Databricks notebook source
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS account;
# MAGIC create table account
# MAGIC (account_id float,
# MAGIC district_id float,
# MAGIC frequency string,
# MAGIC `date` float)
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/account.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS client;
# MAGIC create table client
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/client.asc",  header "true", delimiter ";", inferSchema "true")
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS disp;
# MAGIC create table disp
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/disp.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS district;
# MAGIC create table district
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/district.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS loan;
# MAGIC create table loan
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/loan.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS trans;
# MAGIC create table trans
# MAGIC USING csv
# MAGIC OPTIONS (path "/FileStore/curso/trans.asc",  header "true", delimiter ";", inferSchema "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *
# MAGIC from trans

# COMMAND ----------

# MAGIC %sql
# MAGIC -- resumen de las transacciones por cuenta
# MAGIC DROP TABLE IF EXISTS r_trans;
# MAGIC CREATE TABLE r_trans
# MAGIC AS
# MAGIC SELECT 
# MAGIC     account_id,
# MAGIC     MAX(operation) AS operation,
# MAGIC     AVG(amount) AS monto_dinero_promedio,
# MAGIC     AVG(balance) AS balance_promedio,
# MAGIC     COUNT(*) AS cantidad_transacciones
# MAGIC FROM trans
# MAGIC GROUP BY account_id
# MAGIC ;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * 
# MAGIC from r_trans
# MAGIC limit 10
# MAGIC ;

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC -------------------------------------------------
# MAGIC -- CREACION DE LA TABLA MINABLE DESNORMALIZADA 
# MAGIC -------------------------------------------------
# MAGIC
# MAGIC CREATE OR REPLACE view tabla_minable 
# MAGIC as
# MAGIC SELECT *
# MAGIC FROM 
# MAGIC (
# MAGIC 	SELECT client_id, district_id
# MAGIC 	     , if (SUBSTRING(birth_number, 3, 2) <= '50', 'hombre', 'mujer') as genero -- extrayendo el genero
# MAGIC     FROM 
# MAGIC 	    client
# MAGIC ) as c
# MAGIC 	LEFT JOIN (
# MAGIC 				SELECT A1 as districtid
# MAGIC 					 , A2 as distrito
# MAGIC 					 , A3 as region
# MAGIC 					 , A4 as habitantes
# MAGIC 					 , A11 as salario_promedio
# MAGIC 				FROM 
# MAGIC 					district
# MAGIC              	) as dis on dis.districtid = c.district_id
# MAGIC   	INNER JOIN (
# MAGIC 				SELECT client_id as c_id , account_id, type as tipo_disposicion_cliente
# MAGIC                 FROM
# MAGIC 					disp
# MAGIC 				) as di on di.c_id = c.client_id
# MAGIC   	INNER JOIN (
# MAGIC 				SELECT account_id as ac_id
# MAGIC 					 -- , district_id as district_id_cuenta -- info  a consideracion debido a que nos interesa el distrito del cliente
# MAGIC 					 -- , STR_TO_DATE(date, '%y%m%d') as fecha_creacion_cuenta -- conversion de la fecha
# MAGIC 					 , case
# MAGIC 						  when frequency = 'POPLATEK MESICNE' then 'emision mensual'
# MAGIC 						  when frequency = 'POPLATEK TYDNE' then 'emision semanal'
# MAGIC 						  when frequency = 'POPLATEK PO OBRATU' then 'emision despues de una transaccion'
# MAGIC 					    else 'no especifica'
# MAGIC 					    end as frecuencia_emision			
# MAGIC 				FROM 
# MAGIC 					account
# MAGIC 				) as ac on ac.ac_id = di.account_id
# MAGIC  	RIGHT JOIN ( -- Obtener solo los que tuvieron prestamos
# MAGIC 				SELECT loan_id, account_id as acl_id
# MAGIC 					 , amount as monto_prestamo, duration as duracion_prestamo
# MAGIC 					 , payments AS pagos_mensuales
# MAGIC                      , status  as estado
# MAGIC 				FROM 
# MAGIC 					loan
# MAGIC              	) as ln on ln.acl_id = ac.ac_id
# MAGIC 	LEFT JOIN (	-- Obtener las transacciones realizadas de las personas
# MAGIC 				-- que han obtenido un credito
# MAGIC 				SELECT 
# MAGIC 				account_id as act_id,
# MAGIC 				operation, 
# MAGIC 				monto_dinero_promedio,
# MAGIC     			balance_promedio,
# MAGIC     			cantidad_transacciones
# MAGIC 				FROM 
# MAGIC 					r_trans
# MAGIC              	) as rt on rt.act_id = ac.ac_id
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tabla_minable

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE tabla_minable;

# COMMAND ----------

# MAGIC %sql
# MAGIC select distrito, region, habitantes, salario_promedio
# MAGIC      , frecuencia_emision, monto_prestamo, duracion_prestamo
# MAGIC      , pagos_mensuales, operation, monto_dinero_promedio
# MAGIC      , balance_promedio, cantidad_transacciones
# MAGIC      , estado 
# MAGIC from 
# MAGIC     tabla_minable
# MAGIC ;

# COMMAND ----------

df = spark.sql(""" 
                select distrito, region, habitantes, salario_promedio
                    , frecuencia_emision, monto_prestamo, duracion_prestamo
                    , pagos_mensuales, operation, monto_dinero_promedio
                    , balance_promedio, cantidad_transacciones
                    , estado 
                from 
                    tabla_minable
                """
                )

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_clean = df.na.drop()
df_clean.show()

# COMMAND ----------


df_clean.count()

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df_clean.groupBy(F.col('estado')).count().show()

# COMMAND ----------


# Etiquetas de candidatos
from pyspark.sql.functions import when, col
df_clean = df_clean.withColumn(
    'candidato',
    when((df_clean['estado'] == 'A') | (df_clean['estado'] == 'C'), 'Candidato')
    .otherwise('No candidato')
)
df_clean.show()

# COMMAND ----------

display(df_clean)

# COMMAND ----------

df_final = df_clean.drop('estado')
display(df_final)

# COMMAND ----------

df_final.groupBy(F.col('candidato')).count().show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# one hot encoding distrito
district_indexer = StringIndexer(inputCol= 'distrito', outputCol= 'districtIndex')
district_encoder = OneHotEncoder(inputCol = 'districtIndex', outputCol= 'districtVec')

# one hot encoding region
region_indexer = StringIndexer(inputCol='region', outputCol= 'regionIndex')
region_encoder = OneHotEncoder(inputCol = 'regionIndex', outputCol= 'regionVec')

# one hot encoding frecuencia_emision
frecuencia_indexer = StringIndexer(inputCol='frecuencia_emision', outputCol= 'frecuenciaIndex')
frecuencia_encoder = OneHotEncoder(inputCol = 'frecuenciaIndex', outputCol= 'frecuenciaVec')

# one hot encoding tipo
operation_indexer = StringIndexer(inputCol='operation', outputCol= 'operationIndex')
operation_encoder = OneHotEncoder(inputCol = 'operationIndex', outputCol= 'operationVec')


# COMMAND ----------

df_final.columns

# COMMAND ----------

candidato_indexer = StringIndexer(inputCol= 'candidato', outputCol= 'candidatoIndex')

assembler = VectorAssembler(inputCols = ['districtVec', 'regionVec', 'frecuenciaVec', 'operationVec',
                                        'habitantes',
                                        'salario_promedio',
                                        'monto_prestamo',
                                        'duracion_prestamo',
                                        'pagos_mensuales',
                                        'monto_dinero_promedio',
                                        'balance_promedio',
                                        'cantidad_transacciones',
                                        ], outputCol= 'features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

log_reg_candidato = LogisticRegression(featuresCol= 'features', labelCol='candidatoIndex')

# COMMAND ----------

pipeline = Pipeline(stages= [
    district_indexer,
    region_indexer,
    frecuencia_indexer,
    operation_indexer,
    district_encoder,
    region_encoder,
    frecuencia_encoder,
    operation_encoder,
    candidato_indexer,
    assembler, 
    log_reg_candidato])

# COMMAND ----------

train_data, test_data = df_final.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

type(fit_model)

# COMMAND ----------

results = fit_model.transform(test_data)
type(results)

# COMMAND ----------

display(results)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results.select('candidatoIndex', 'prediction').show(10)

# COMMAND ----------

auc= me_eval.evaluate(results)
print(auc)

# COMMAND ----------

training_summary = fit_model.stages[-1].summary
roc = training_summary.roc.toPandas()

import matplotlib.pyplot as plt
plt.plot(roc['FPR'], roc['TPR'])
plt.ylabel('True positive rate')
plt.xlabel('False postive rate')
plt.title('curva ROC')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient-Boosted Trees (GBTs)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

model = GBTClassifier(featuresCol= 'features', labelCol='candidatoIndex')

# COMMAND ----------

pipeline = Pipeline(stages= [
    district_indexer,
    region_indexer,
    frecuencia_indexer,
    operation_indexer,
    district_encoder,
    region_encoder,
    frecuencia_encoder,
    operation_encoder,
    candidato_indexer,
    assembler, 
    model])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

results= fit_model.transform(test_data)
display(results)

# COMMAND ----------

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')

results.select('candidatoIndex', 'prediction').show(10)

# COMMAND ----------

auc= me_eval.evaluate(results)
print(auc)

# COMMAND ----------

model.uid

# COMMAND ----------

def train_model(modelo, features='features', labelcol='candidatoIndex'):
    model = modelo(featuresCol=features, labelCol=labelcol)
    pipeline = Pipeline(stages= [
                                district_indexer,
                                region_indexer,
                                frecuencia_indexer,
                                operation_indexer,
                                district_encoder,
                                region_encoder,
                                frecuencia_encoder,
                                operation_encoder,
                                candidato_indexer,
                                assembler, 
                                model])
    fit_model = pipeline.fit(train_data)
    results= fit_model.transform(test_data)
    me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = labelcol)
    auc= me_eval.evaluate(results)
    print(f'{model.uid}-- > AUC: {auc}')
    return fit_model

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier, NaiveBayes, LinearSVC

gbt_model = train_model(GBTClassifier)
nb_model = train_model(NaiveBayes)
svc_model = train_model(LinearSVC)


# COMMAND ----------

feature_importances = gbt_model.stages[-1].featureImportances

# COMMAND ----------

# Supongamos que tienes una lista de nombres de características que corresponden a tus datos.
feature_names = ["feature1", "feature2", "feature3", ...]  # Reemplaza con tus nombres de características.

# Crea una lista de índices para ordenar las importancias de las características.
indices = range(len(feature_importances))

# Ordena las importancias de las características en orden descendente.
sorted_feature_importances = sorted(zip(feature_importances, feature_names), reverse=True)

# Extrae las importancias ordenadas y los nombres de las características correspondientes.
sorted_importances, sorted_names = zip(*sorted_feature_importances)

# Crea el gráfico de barras.
plt.figure(figsize=(10, 6))
plt.bar(indices, sorted_importances, align="center")
plt.xticks(indices, sorted_names, rotation=90)
plt.xlabel("Características")
plt.ylabel("Importancia")
plt.title("Importancia de las Características")
plt.tight_layout()

# Muestra el gráfico.
plt.show()
