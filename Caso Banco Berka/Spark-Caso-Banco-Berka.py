# Databricks notebook source
# MAGIC %md
# MAGIC # Banco Berka - ML

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creación y transformación de las tablas

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS account;
# MAGIC create table account
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
# MAGIC select * 
# MAGIC from tabla_minable
# MAGIC ;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE tabla_minable;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selección de columnas finales

# COMMAND ----------

# MAGIC %sql
# MAGIC select region, habitantes, salario_promedio
# MAGIC      , frecuencia_emision, monto_prestamo, duracion_prestamo
# MAGIC      , pagos_mensuales, operation, monto_dinero_promedio
# MAGIC      , balance_promedio, cantidad_transacciones
# MAGIC      , estado 
# MAGIC from 
# MAGIC     tabla_minable
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tabla minable - SQL a Dataframe Spark

# COMMAND ----------

df = spark.sql(""" 
                select region, habitantes, salario_promedio
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

# limpieza de valores nulos
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

df_final.printSchema()

# COMMAND ----------

# Division de los datos en train 70% y test 30%
train_data, test_data = df_final.randomSplit([0.7,0.3])

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# one hot encoding region
region_indexer = StringIndexer(inputCol='region', outputCol= 'regionIndex', handleInvalid='skip')
region_encoder = OneHotEncoder(inputCol = 'regionIndex', outputCol= 'regionVec')

# one hot encoding frecuencia_emision
frecuencia_indexer = StringIndexer(inputCol='frecuencia_emision', outputCol= 'frecuenciaIndex', handleInvalid='skip')
frecuencia_encoder = OneHotEncoder(inputCol = 'frecuenciaIndex', outputCol= 'frecuenciaVec')
# one hot encoding tipo
operation_indexer = StringIndexer(inputCol='operation', outputCol= 'operationIndex', handleInvalid='skip')
operation_encoder = OneHotEncoder(inputCol = 'operationIndex', outputCol= 'operationVec')


candidato_indexer = StringIndexer(inputCol= 'candidato', outputCol= 'candidatoIndex')

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['regionVec', 
                                        'frecuenciaVec', 
                                        'operationVec',
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

# MAGIC %md
# MAGIC ## Entrenamiento de modelos

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelo inicial base

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

log_reg_candidato = LogisticRegression(featuresCol= 'features', labelCol='candidatoIndex')

# COMMAND ----------

pipeline = Pipeline(stages= [
    region_indexer,
    frecuencia_indexer,
    operation_indexer,
    region_encoder,
    frecuencia_encoder,
    operation_encoder,
    candidato_indexer,
    assembler, 
    log_reg_candidato])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

type(fit_model)

# COMMAND ----------

results = fit_model.transform(test_data)
type(results)

# COMMAND ----------

results.show()

# COMMAND ----------

display(results)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results.select('candidatoIndex', 'prediction', 'probability').show(10)

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
# MAGIC ## Entrenamiento de otros modelos y comparación

# COMMAND ----------

# Funcion para entrenar distintos modelos 
def train_model(modelo):
    pipeline_model = Pipeline(stages= [
                                region_indexer,
                                frecuencia_indexer,
                                operation_indexer,
                                region_encoder,
                                frecuencia_encoder,
                                operation_encoder,
                                candidato_indexer,
                                assembler, 
                                modelo])
    evals = []
    auc_ini=0
    for i in range(3):
        train_data, test_data = df_final.randomSplit([0.7,0.3])
        fit_model = pipeline_model.fit(train_data)
        results = fit_model.transform(test_data)
        me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
        auc= me_eval.evaluate(results)
        evals.append(auc)
        print(f'{modelo.uid} --- > AUC: {auc}')
        if auc > auc_ini:
            auc_ini = auc
            best_fit_model = fit_model
            results_model = results 

    media_auc = sum(evals) / len(evals)
    print(f'{modelo.uid} --- > AUC_prom: {media_auc}')
    return best_fit_model, results_model

# COMMAND ----------

# Modelos
from pyspark.ml.classification import GBTClassifier, LogisticRegression, DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ### GBTClassifier

# COMMAND ----------

gbt = GBTClassifier(featuresCol= 'features', labelCol='candidatoIndex',
                    maxIter=25, maxDepth=6, maxBins=16)
gbt_fit_pipe, results_gbt = train_model(gbt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluación GBT
# MAGIC

# COMMAND ----------

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results_gbt.select('candidatoIndex', 'prediction', 'probability').show(10)

# COMMAND ----------

auc= me_eval.evaluate(results_gbt)
print(auc)

# COMMAND ----------

feature_importances = gbt_fit_pipe.stages[-1].featureImportances

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Índice de la característica')
plt.ylabel('Importancia de las características')
plt.title('Feature Importance del modelo GBTClassifier')
plt.xticks(range(len(feature_importances)), range(len(feature_importances)))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arbol de decision

# COMMAND ----------

dt = DecisionTreeClassifier(featuresCol= 'features', labelCol='candidatoIndex',
                            maxDepth=10, maxBins=32)
dt_fit_pipe, results_dt = train_model(dt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluación Arbol de desición

# COMMAND ----------

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results_dt.select('candidatoIndex', 'prediction', 'probability').show(10)

# COMMAND ----------

auc= me_eval.evaluate(results_dt)
print(auc)

# COMMAND ----------

feature_importances = dt_fit_pipe.stages[-1].featureImportances

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Índice de la característica')
plt.ylabel('Importancia de las características')
plt.title('Feature Importance del modelo GBTClassifier')
plt.xticks(range(len(feature_importances)), range(len(feature_importances)))
plt.tight_layout()
plt.show()

# COMMAND ----------

train_data.columns[:-1]

# COMMAND ----------

tree_debug_string = dt_fit_pipe.stages[-1].toDebugString
print(tree_debug_string)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regresion Logistica

# COMMAND ----------

lr = LogisticRegression(featuresCol= 'features', labelCol='candidatoIndex')
lr_fit_pipe, results_lr = train_model(lr)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluación - Regresión logistica

# COMMAND ----------

me_eval = BinaryClassificationEvaluator(rawPredictionCol= 'prediction', labelCol = 'candidatoIndex')
results_lr.select('candidatoIndex', 'prediction', 'probability').show(10)

# COMMAND ----------

auc= me_eval.evaluate(results_lr)
print(auc)

# COMMAND ----------

training_summary = lr_fit_pipe.stages[-1].summary
roc = training_summary.roc.toPandas()

import matplotlib.pyplot as plt
plt.plot(roc['FPR'], roc['TPR'])
plt.ylabel('True positive rate')
plt.xlabel('False postive rate')
plt.title('curva ROC')
plt.show()
