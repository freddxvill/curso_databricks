# Databricks notebook source
# MAGIC %md
# MAGIC Pasageros que viajan mensualmente en miles por avion desde enero 1949 hasta diciembre 1960

# COMMAND ----------

import pandas as pd
import numpy as np
%matplotlib inline

# COMMAND ----------

airline = pd.read_csv("drive/MyDrive/CursoSeriesTiempo/datos/airline_passengers.csv", index_col='Month', parse_dates=True)

# COMMAND ----------

airline.head()

# COMMAND ----------

airline.plot()

# COMMAND ----------

from statsmodels.tsa.seasonal import seasonal_decompose

# COMMAND ----------

result = seasonal_decompose(airline['Thousands of Passengers'], model='additive')
result.plot();

##  Se usa cuando la tendencia es detipo lineal se usa el modelo aditivo

# COMMAND ----------

result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
result.plot();

## Cuando es exponencial es mejor que sea multiplicativo, en este caso se usa el modelo multiplicativo

# COMMAND ----------


