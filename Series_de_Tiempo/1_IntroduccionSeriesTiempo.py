# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

raw_data = pd.read_csv("/dbfs/FileStore/Index2018.csv")

# COMMAND ----------

# para tener los datos originales
df_comp = raw_data.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Examinar datos
# MAGIC

# COMMAND ----------

df_comp.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - Date - Dia en el que el valor fue registrado
# MAGIC - spx  - SP500 Bolsa de Estados Unidos 
# MAGIC - dax  - Dax40 Bolsa Alemana  
# MAGIC - ftse - ftse100 Bolsa de Londres
# MAGIC - nikkei - Nikkei225  Bolsa de Japon
# MAGIC
# MAGIC Los valores son datos de series temporales para los precios de cierre de 4 indices de mercado 
# MAGIC Cada indice de mercado es una cartera de las empresas publicas mas cotizadas en los mercados bursatiles
# MAGIC
# MAGIC

# COMMAND ----------

df_comp

# COMMAND ----------

df_comp.describe()

# COMMAND ----------

# is not available?
df_comp.isna()

# COMMAND ----------

df_comp.isna().sum()

# COMMAND ----------

df_comp.spx.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Graficando los datos

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

df_comp.spx.plot(figsize=(22,5), title = "S&P500 precios")
plt.show()
# Se muestran números en el eje X debido a que se esta mostrando el indice del objeto Pandas

# COMMAND ----------

df_comp.dax.plot(figsize=(22,5), title = "DAX precios")
plt.show()

# COMMAND ----------

df_comp.ftse.plot(figsize=(22,5), title="FTSE100 prices")
plt.show()

# COMMAND ----------

# USA y Reino Unido son similares, por que no colocarlas en un solo grafico
df_comp.spx.plot(figsize=(22,5), title="SP500 Prices")
df_comp.ftse.plot(figsize=(22,5), title="FTSE100 prices")
plt.title("SP versus FTSE")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Densidad de datos, que probabilidad existe para cada dato
# MAGIC Grafico QQ Quantile - Quantile
# MAGIC
# MAGIC La grafica indica si los datos se aproxima a una curva normal

# COMMAND ----------

import scipy.stats

# COMMAND ----------

scipy.stats.probplot(df_comp.spx, plot =plt)
plt.title("QQ plot", size = 24)
plt.show()

# en X la cantidad de desviaciones estandar al rededor de la media
# en Y los valores de la variable ordenados de mejor a mayor
# en este caso los datos no se distribuyen normalmente por lo que no se puede usar una regresión u otro metodo que se base en ciertas suposciones de normalidad

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversión del dataset Pandas para ser analizado por series de tiempo 
# MAGIC
# MAGIC ### Configurar el indice

# COMMAND ----------

# de texto a Fecha
# dayfirst=True indica que el formato del texto fecha empieza con el día
df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)

# COMMAND ----------

df_comp.head()

# COMMAND ----------

df_comp.date.describe()

# COMMAND ----------

# configurar la variable fecha (date) como indice, inplace para que la misma variable cambie  sea indice definitivamente
df_comp.set_index("date", inplace=True)

# COMMAND ----------

df_comp.head()

# COMMAND ----------

df_comp.date.describe() # correcto que salga error ahora es un indice no una variable

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Configuración de frecuencia**
# MAGIC
# MAGIC Los datos de series de tiempo requieren una frecuencia constante

# COMMAND ----------

# Configurar la frecuencia deseada, esto es muy importante!
# h hora w semanal d diario m mes a anual b business_days
df_comp=df_comp.asfreq('d')


# COMMAND ----------

# Aparece 8 y 9 ya que son fines de semana
df_comp.head()

# COMMAND ----------

# para evitar este problema se puede definir la frecuencia como dias de trabajo
df_comp=df_comp.asfreq('b')

# COMMAND ----------

df_comp.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manejo de datos faltantes

# COMMAND ----------

# completar valores faltantes, esto se hace asi ya que cambiamos las propiedades de los datos que son series de tiempo
df_comp.isna()


# COMMAND ----------

# hay 8 valores faltantes
df_comp.isna().sum()

# COMMAND ----------

# 1ra forma
df_comp.spx=df_comp.spx.fillna(method='ffill')  # front filling - relleno frontal, le asigna el valor del periodo posterior back filling es periodo anterior

# COMMAND ----------

df_comp.isna().sum()

# COMMAND ----------

# 2da forma
df_comp.ftse=df_comp.spx.fillna(method='bfill') # back filling - Periodo anterior

# COMMAND ----------

#tambien se puede llenar con valores constantes, ejemplo 1 media per oesto noo es aconsejable,solo en series estacionarias podria tener sentido pero en series temporales no tiene sentido
df_comp.dax=df_comp.dax.fillna(df_comp.dax.mean())

# COMMAND ----------

df_comp.nikkei=df_comp.nikkei.fillna(method='bfill')

# COMMAND ----------

df_comp.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reducir o simplificar el dataset
# MAGIC
# MAGIC Solo una columna que es la que se analizara

# COMMAND ----------

## borrar o agregar columnas

# COMMAND ----------

# crear columna adicion con un nombre generico para la variable spx
df_comp['market_value'] = df_comp.spx

# COMMAND ----------

df_comp.describe()

# COMMAND ----------

# borrar una columna
del df_comp['spx']


# COMMAND ----------

df_comp.describe()

# COMMAND ----------

del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']


# COMMAND ----------

# solo nos quedamos con el SP500
df_comp.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### **División de datos para aprendizaje automático**

# COMMAND ----------

# division de datos de entrenamiento y validación
# no se puede sacar muestras aleatorias, entrenamiento validación debe hacerse de manera secuencial
# los criterios de distribución porcentual es igual que en los clasicos modelos
size = int(len(df_comp)*0.8)

# COMMAND ----------

size

# COMMAND ----------

df = df_comp.iloc[:size]

# COMMAND ----------

df_test = df_comp.iloc[size:]

# COMMAND ----------

# comprobar que no existan datos solapados

df.tail()

# COMMAND ----------

df_test.head()
