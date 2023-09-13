# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Preprocesamiento de datos y Feature Engineering
# MAGIC

# COMMAND ----------

import pandas as pd
 
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

white_wine

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

# COMMAND ----------

white_wine

# COMMAND ----------

red_wine

# COMMAND ----------

data = pd.concat([red_wine, white_wine], axis=0)

# COMMAND ----------

data

# COMMAND ----------

data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data

# COMMAND ----------

data.head()

# COMMAND ----------

data.columns

# COMMAND ----------

data.dtypes

# COMMAND ----------

data.shape


# COMMAND ----------

data.corr()

# COMMAND ----------

data.describe()

# COMMAND ----------

import seaborn as sns
sns.displot(data.quality)

# COMMAND ----------

# Creación de una variable binaria en función si se cumple o no con la condición de calidad mayor o igual que 7
high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

(data.quality >= 7).astype(int)

# COMMAND ----------

data

# COMMAND ----------

import matplotlib.pyplot as plt
 
dims = (3, 4)
 
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots no pueden ser usados en variables dicotomicas
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

data.isna().any()

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
X = data.drop(["quality"], axis=1)
y = data.quality
 
# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
 
# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md
# MAGIC # Parte 2: Construccion del modelo base

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

# COMMAND ----------

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
 
 
with mlflow.start_run(run_name='random_forest_no_tuneado2'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)
 
  # predict_proba retorna [prob_negativa, prob_positiva], asi que nos quedamos con la salida de [:, 1] la probabilidad positiva
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  # inicializa un log para el parametro n_estimators
  mlflow.log_param('n_estimators', n_estimators)
 
  # Usa el área bajo la curba ROC como una métrica
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
 
  # Log the model with a signature that defines the schema of the model's inputs and outputs. When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models. The necessary dependencies are added to a conda.yaml file which is logged along with the model
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "random_forest_no_tuneado2"').iloc[0].run_id
print(run_id)

# COMMAND ----------

model_name = "wine_quality2"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
 
# El registro del modelo puede tomar unos cuantos segundos, por eso se añade un tiempo de espera
time.sleep(15)

# COMMAND ----------

from mlflow.tracking import MlflowClient     #  importa la clase MlflowClient
 
client = MlflowClient()                      # Esta instancia se utilizará para realizar operaciones en el servidor de seguimiento de MLflow
client.transition_model_version_stage(       # ealizar la transición de una versión específica de un modelo a la etapa de Produccion
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
 
# Esta salida deberia ser igual a la salida del modelo en producion
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Experimentar con un nuevo modelo
# MAGIC

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
#  fmin         es una función principal de Hyperopt que se utiliza para realizar la optimización de hiperparámetros
#  tpe          es un algoritmo de optimización de hiperparámetros basado en el procesador de Thompson (Thompson Sampling)
#  hp           Permite especificar los tipos de hiperparámetros (por ejemplo, continuos o discretos) y sus rangos de búsqueda
#  SparkTrials  es una clase que permite realizar optimización de hiperparámetros distribuida en un clúster de Spark
#  Trials       es una clase que se utiliza para realizar un seguimiento de los resultados de las evaluaciones de hiperparámetros durante la optimización
#  STATUS_OK     es una constante que se utiliza para indicar que la evaluación de un conjunto de hiperparámetros tuvo éxito

from hyperopt.pyll import scope   #  es una función que se utiliza para definir espacio de búsqueda
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

# COMMAND ----------

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # configurar una semilla para cada entrenamiento
}

# COMMAND ----------

def train_model(params):
  # con MLFLow autologging, los hiperparámetros y el modelo entrenado son automaticamente logeados a MLFLow
  
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)
    # Pase el conjunto de validación para que xgb pueda realizar un seguimiento de una métrica de evaluación. XGBoost 
    # finaliza el entrenamiento cuando la métrica de evaluación ya no mejora.

    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)
 
    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Configurar la perdida para que -1*auc_score asi fmin maximiza el valor de auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}

# COMMAND ----------

#  En este caso, algo se está configurando para utilizar el algoritmo de optimización de "Tree-structured Parzen Estimator" (TPE)
algo = tpe.suggest

# COMMAND ----------

from hyperopt import SparkTrials
 
# Un mayor paralelismo conducirá a aceleraciones, pero a un barrido de hiperparámetros menos óptimo. 
# Un valor razonable para el paralelismo es la raíz cuadrada de max_evals.

spark_trials = SparkTrials(parallelism=10)

# COMMAND ----------

with mlflow.start_run(run_name='xgboost_models2'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=algo, 
    max_evals=12,
    trials=spark_trials,
  )

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)
 
# Registering the model takes a few seconds, so add a small delay
time.sleep(15)

# COMMAND ----------

# Para colocar el produccion el nuevo modelo
# Archivar la version del modelo anterior
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)
 
# Promover la nueva versión del modelo a produccion
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 4: Inferencia en Batch
# MAGIC

# COMMAND ----------

spark_df = spark.createDataFrame(X_train)
# Reemplazar <username> con su username antes de correr esta celda.
table_path = "dbfs:/dannyhuanca@hotmail.com/delta/wine_data"
# Borrar el contenido de este path en caso esta celda ya haya sido corrida.
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

import mlflow.pyfunc
apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Leer los nuevos datos desde la tabla Delta.
new_data = spark.read.format("delta").load(table_path)

# COMMAND ----------

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct
 
#La función struct se utiliza para crear una nueva columna que contiene una estructura (estructura de datos) a partir de una lista de columnas existentes en un DataFrame de PySpark

# Aplicar el modelo a los nuevos datos

# El operador * se utiliza para desempaquetar la lista de nombres de columnas como argumentos separados para la función struct. Esto crea una estructura que contiene todas las columnas de X_train

udf_inputs = struct(*(X_train.columns.tolist()))
 
new_data = new_data.withColumn(
  "prediccion2",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 5: Modelo en modo servicio

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapi78f9ga8381asd2d43f4497asdfad22f4b08e65fc-3"

# COMMAND ----------

    import os
    import requests
    import numpy as np
    import pandas as pd
    import json

    def create_tf_serving_json(data):
      return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

    def score_model(dataset):
      url = 'https://adb-39178re66513f57r48740.0.azuredatabricks.net/serving-endpoints/Prueba_wine_quality/invocations'
      headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
    'Content-Type': 'application/json'}
      ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
      data_json = json.dumps(ds_dict, allow_nan=True)
      response = requests.request(method='POST', headers=headers, url=url, data=data_json)
      if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
      return response.json()

# COMMAND ----------

num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])

# COMMAND ----------

print(type(served_predictions))
print(type(model_evaluations))

# COMMAND ----------

served_predictions_df = pd.DataFrame.from_dict(served_predictions, orient='columns')
model_evaluations_df = pd.DataFrame(model_evaluations)

# COMMAND ----------

result = pd.concat([model_evaluations_df, served_predictions_df], axis=1)
result.columns = ["Model Prediction", "Served Model Prediction"]
print(result)

# COMMAND ----------


