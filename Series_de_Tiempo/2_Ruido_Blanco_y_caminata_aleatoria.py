# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()

# COMMAND ----------

raw_csv_data = pd.read_csv("/dbfs/FileStore/Index2018.csv")

# COMMAND ----------

df_comp = raw_csv_data.copy()

# COMMAND ----------

df_comp.head()

# COMMAND ----------

df_comp.date = pd.to_datetime(df_comp.date, dayfirst=True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')

# COMMAND ----------

df_comp['market_value'] = df_comp.spx

# COMMAND ----------

del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

# COMMAND ----------

# Ruido Blanco
# loc medida de tendencia central
# scale medida de dispersión
# size tamaño
wn =np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size=len(df))

# COMMAND ----------

# añadir la nueva variable al dataset original
df['wn'] = wn

# COMMAND ----------

df.describe()


# COMMAND ----------

df.wn.plot(figsize = (22,5))
plt.title("Ruido blanco en series de tiempo", size = 24)
plt.show()

# COMMAND ----------

df.market_value.plot(figsize = (22,5))
plt.title("SP prices", size = 24)
plt.ylim(0, 2300)  # para que tengan una escala parecida
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Caminata aleatoria
# MAGIC Si una serie tiene una estructura en la que el valor del futuro cambia aleatoriamente entonces esta no es estacionaria

# COMMAND ----------

rw = pd.read_csv("/dbfs/FileStore/RandWalk.csv")

# COMMAND ----------

rw.head()

# COMMAND ----------

rw.date = pd.to_datetime(rw.date, dayfirst = True)
rw.set_index("date", inplace = True)
rw.head()

# COMMAND ----------

rw = rw.asfreq('b')
rw.describe()

# COMMAND ----------

# al Datafrae df que ya contiene la serie del indicador ademas de la serie de ruido blanco, se le añada esta columna de caminata aleatoria

df['rw'] = rw.price
df.head()

# COMMAND ----------

# Como esta en la teoria es el precio anterior mas un valor aleatorio
df.rw.plot(figsize = (20,5))
plt.title("Random Walk", size = 24)
plt.show()

# COMMAND ----------

# la caminada aletaria vs la variable original
df.rw.plot(figsize = (20,5))
df.market_value.plot()
plt.title("Random Walk vs S&P", size = 24)
plt.legend()
plt.show()

# COMMAND ----------

# generación de caminata aleatoria
steps = np.random.normal(loc=0, scale=1, size=500)
steps[0]=0
p = 100 + np.cumsum(steps)

# COMMAND ----------

plt.plot(p)
plt.title("Caminata aleatoria")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Test de Dickey-Fuller para determinar Estacionariedad**

# COMMAND ----------

# Test de Dickey -Fuller -- Estacionariedad
sts.adfuller(df.market_value)

# -1.73  valor de estadistico
#  los valores criticos son 1%, 10%, 5%
#  el p-vallue es 0.41  como es 0.41 > 0.05 por lo que NO SE RECHAZA la hipótesis nula por lo que no es estacionaria
#  18 es el número de retrasos hay autocorrelación a 18 periodos atras

# COMMAND ----------

# Ruido blando es estacionario para comprobarlo
sts.adfuller(df.wn)
# rechazamos la hipotesis nula 0 < 0.05

# COMMAND ----------

# para el caso de la caminata aleatoria
sts.adfuller(df.rw)
#  0.61 no se rechaza por lo que no es estacionaria 0.61 > 0.05

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Estacionalidad**

# COMMAND ----------

# Estacionalidad
# no hay un patron ciclico
# En los residuos se observa los efectos de mayor osilación el 2000 y el problema de la burbuja del 2008

s_dec_aditivo = seasonal_decompose(df.market_value, model="additive")
s_dec_aditivo.plot()
plt.show()



# COMMAND ----------

s_dec_multiplicativo = seasonal_decompose(df.market_value, model="multiplicative")
s_dec_multiplicativo.plot()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # AFC

# COMMAND ----------

# ACF
# normalmente se define como 40 la cantidad de desfaces a calcular
# no se identifica la primera auto correlación que seria la misma serie, por lo que siempre esta es igual a 1

sgt.plot_acf(df.market_value, lags = 40, zero = False)
plt.title("ACF S&P", size = 24)
plt.show()


# el area azul son la significancia, esta se expande a mas distancia en el tiempo la correlación es menos probable
# todas las lineas son mas altas que la region azul, indica que la autocorrelación es significativa
# la autocorrelaión a penas disminuye, lo que sugiere que 40 dias atras los precios se mantiene

# COMMAND ----------

sgt.plot_acf(df.wn, lags = 40, zero = False)
plt.title("ACF SP", size = 24)
plt.show()

# completamente diferentes, los valores camian tanto positivos como negativos, todas las lineas caen en la región azul, por lo que no son signifiativos
# no hay autocorrelación

# COMMAND ----------

sgt.plot_acf(df.rw, lags = 40, zero = False)
plt.title("ACF SP", size = 24)
plt.show()

# se parece mas a los precios

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Autocorrelación Parcial**

# COMMAND ----------

# Autocorrelación parcial

sgt.plot_pacf(df.market_value, lags = 40, zero = False, method = ('ols'))  # ols = ordinary least squares
plt.title("PAFC SP", size = 24)
plt.show()

# se tienen en azul el area de significación los primeros solamente son difernetes de cero. Solo los primeros son significativos
# se puede observar que existen valores que son negativos, por lo que implican una relación negativa.
# la ventaja de esta forma de analisis es que elimina los efectos intermedios o los indirectos que puedan influir en la autocorrelación

# COMMAND ----------

sgt.plot_pacf(df.wn, lags = 40, zero = False, method = ('ols'))
plt.title("PAFC SP", size = 24)
plt.show()

# COMMAND ----------

sgt.plot_pacf(df.rw, lags = 40, zero = False, method = ('ols'))
plt.title("PAFC SP", size = 24)
plt.show()

# COMMAND ----------


