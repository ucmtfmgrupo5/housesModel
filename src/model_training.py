# # CARGA DE LIBRERÍAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import os
import random
import re
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from math import e
from unicodedata import normalize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
pd.options.display.max_rows = 300
pd.options.display.max_columns = None
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# # CARGA DE DATOS

# Carga del CSV y creación del dataframe 
url_houses_madrid="https://raw.githubusercontent.com/ucmtfmgrupo5/database/main/houses_Madrid_v3.csv"
s=requests.get(url_houses_madrid).content
df_vivienda=pd.read_csv(io.StringIO(s.decode("latin-1")), sep=";")

# Cabecera del dataframe creado
df_vivienda.head(5)


# # INSPECCIÓN DE LOS DATOS


df_vivienda.describe()



#Datos missing 
total = df_vivienda.isnull().sum().sort_values(ascending=False)
porcentaje = (df_vivienda.isnull().sum()/df_vivienda.isnull().count()).sort_values(ascending=False)
datos_missing = pd.concat([total, porcentaje], axis=1, keys=['Total', 'Porcentaje'])
datos_missing


lista_categoricas=['n_rooms','n_bathrooms', 'is_exact_address_hidden', 'floor', 'is_floor_under',
       'operation', 'house_type_id', 'is_renewal_needed', 'is_new_development',
       'has_central_heating', 'has_individual_heating', 'has_ac',
       'has_fitted_wardrobes', 'has_lift', 'is_exterior', 'has_garden',
       'has_pool', 'has_terrace', 'has_balcony', 'has_storage_room',
       'is_accessible', 'has_green_zones', 'energy_certificate', 'has_parking',
       'is_parking_included_in_price', 'is_orientation_north',
       'is_orientation_west', 'is_orientation_south', 'is_orientation_east']

lista_numericas=['sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms', 'rent_price', 'buy_price', 'buy_price_by_area',
       'built_year','parking_price', 'precio_m2_barrio']


for i in lista_categoricas:
    print(df_vivienda[i].value_counts())


# Las conclusiones preliminare que podemos sacar, sobre el estado inicial de los datos es el siguiente:
# 
# 1.- Variables susceptibles de ser consideradas como continuas o como categóricas.
# 2.- Datos missing, en este caso hay datos que parece lógico que se pueda asumir, en las variables dicotómicas los missing serán considerados como FALSE.
# 3.- Datos outliers, existen datos que pueden distorsionar el análisis bien porque son negativos cuando deberías en 0 o positivos o bien porque su escala no corresponde a la distribución de la variable.
# 4.- Existen columnas con información valiosa que conviene descomponer.
# 5.- Posibilidad de extraer coordenadas a partir de direcciones.
# 6.- Datos con escala incorrecta.
# 7.- Eliminar datos redundantes o irrelevantes.
# 8.- Es necesario normalizar nombres de barrios para intentar incluir más información externa.
# 
# ¡Vamos a ello...!

# Eliminamos un 0 de estas variables 



df_vivienda['built_year']=df_vivienda['built_year']/10
df_vivienda['sq_mt_built']=df_vivienda['sq_mt_built']/10
df_vivienda['sq_mt_useful']=df_vivienda['sq_mt_useful']/10
df_vivienda['n_bathrooms']=df_vivienda['n_bathrooms']/10


# A partir del campo 'neighborhood_id', que contiene información del barrio y del distrito, se añaden al dataframe nuevos campos:
# 
# #'barrio_id' #'barrio_nombre' #'precio_m2_barrio' #'distrito_id' #'distrito_nombre' 
# 
# Tras la estracción de datos, se elimina el campo 'neighborhood_id'


# Creación de las nuevas columnas en el dataframe
df_vivienda=df_vivienda.assign(barrio_id="",barrio_nombre="",precio_m2_barrio="",distrito_id="",distrito_nombre="")

# Para cada uno de los registros, se extrae la información y se almacena en el campo correspondiente
for i in range(len(df_vivienda)):
  df_vivienda["barrio_id"][i] = df_vivienda["neighborhood_id"][i][(df_vivienda["neighborhood_id"][i].find('hood')+5):(df_vivienda["neighborhood_id"][i].find(':'))]
  df_vivienda["barrio_nombre"][i] = df_vivienda["neighborhood_id"][i][(df_vivienda["neighborhood_id"][i].find(': ')+2):(df_vivienda["neighborhood_id"][i].find(' ('))]
  df_vivienda["precio_m2_barrio"][i] = df_vivienda["neighborhood_id"][i][(df_vivienda["neighborhood_id"][i].find('(')+1):(df_vivienda["neighborhood_id"][i].find('.'))]
  df_vivienda["distrito_id"][i] = df_vivienda["neighborhood_id"][i][(df_vivienda["neighborhood_id"][i].find('District')+9):(df_vivienda["neighborhood_id"][i].rfind(':'))]
  df_vivienda["distrito_nombre"][i] = df_vivienda["neighborhood_id"][i][(df_vivienda["neighborhood_id"][i].rfind(':')+2):]

# Eliminación del campo 'neighborhood_id', cuya información ya se ha extraido
df_vivienda.drop(['neighborhood_id'], axis=1).head(3)



df_vivienda['precio_m2_barrio'] = pd.to_numeric(df_vivienda['precio_m2_barrio'], errors = 'coerce')



df_vivienda.dtypes


# Los campos 'latitude' y 'longitude' se han cargado utilizando comas en lugar de puntos, por lo que estos campos son tipo string. Para poder manejar la variable como numérica se reemplazan los carácteres coma por punto, y se convierten los campos a tipo numérico (float).



df_vivienda['latitude'] = pd.to_numeric(df_vivienda['latitude'].str.replace(',','.'))
df_vivienda['longitude'] = pd.to_numeric(df_vivienda['longitude'].str.replace(',','.'))





print("Número de nulos en la variable 'latitude':", df_vivienda.latitude.isnull().sum())
print("Número de nulos en la variable 'longitude':", df_vivienda.longitude.isnull().sum())




df_vivienda.describe()


# Hay 5905 registros donde el inmueble no está geocodificado, al carecer de un nombre de calle.
# 
# Para solucionar esto se ha creado un fichero CSV donde, para cada barrio, se incluyen las coordenadas de los extremos de un rectangulo incluido en él.
# 
# longitud_min longitud_max latitud_min latitud_max A partir de este CSV se crea un dataframe, denominado 'df_coordenadas_limites'.



# Carga del CSV y creación del dataframe 
url_coordenadas_limites="https://raw.githubusercontent.com/ucmtfmgrupo5/database/main/limites_barrios.csv"
peticion=requests.get(url_coordenadas_limites).content
df_coordenadas_limites=pd.read_csv(io.StringIO(peticion.decode("iso8859_1")), sep=';',)

# Cabecera del dataframe creado
df_coordenadas_limites.head()


# Para cada uno de los registros del dataframe principal se comprueba si 'street_name' es nulo, y si lo es se calculan unas coordenadas aleatorias comprendidas entre los límites de coordenadas del rectangulo definido para cada barrio. Para ello se usa la función 'uniform' de la librería 'random'.
# 
# De esta manera se evita que el dataframe incluya muchas coordenadas iguales como sucedería si asignamos un único punto por barrio en lugar de un rectángulo suficientemente grande.




for i in range(len(df_vivienda)):
  if pd.isnull(df_vivienda["street_name"][i]):
    barrio = df_vivienda["barrio_nombre"][i]
    df_vivienda["longitude"][i] = random.uniform(df_coordenadas_limites["longitud_min"][df_coordenadas_limites.nombre_barrio == barrio], df_coordenadas_limites["longitud_max"][df_coordenadas_limites.nombre_barrio == barrio])
    df_vivienda["latitude"][i] = random.uniform(df_coordenadas_limites["latitud_min"][df_coordenadas_limites.nombre_barrio == barrio], df_coordenadas_limites["latitud_max"][df_coordenadas_limites.nombre_barrio == barrio])




print("Número de nulos en la variable 'latitude':", df_vivienda.latitude.isnull().sum())
print("Número de nulos en la variable 'longitude':", df_vivienda.longitude.isnull().sum())





for i in range(len(df_vivienda)):
  if pd.notnull(df_vivienda["street_name"][i]):
    calle = df_vivienda["street_name"][i].lower()
    calle = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", normalize( "NFD", calle), 0, re.I)
    calle = normalize( 'NFC', calle)

    df_vivienda["street_name"][i] = calle




df_calles_sin_numero = df_vivienda[(pd.notnull(df_vivienda.street_name))&(pd.isnull(df_vivienda.street_number))][["barrio_nombre","street_name"]]
df_calles_sin_numero.head()




tabla = df_calles_sin_numero.groupby('street_name')['barrio_nombre'].nunique()
tabla.nlargest(25)




#lista_calles_extensas = ['calle de bravo murillo', 'calle del principe de vergara', 'paseo de la castellana', 'calle de alcala', 'urb. si',
#                         'calle del doctor esquerdo', 'calle de arturo soria', 'calle de embajadores', 'urb. arturo soria', 'arturo soria',
#                         'avenida del manzanares', 'bravo murillo', 'calle de toledo', 'paseo de santa maria de la cabeza']

lista_calles_extensas = ['calle de bravo murillo', 'calle del principe de vergara', 'paseo de la castellana', 'calle de alcala', 'urb. si', 
                         'calle del doctor esquerdo', 'calle de arturo soria', 'calle de embajadores', 'urb. arturo soria', 'arturo soria',
                         'avenida del manzanares', 'bravo murillo', 'calle de toledo', 'paseo de santa maria de la cabeza', 'alcala',
                         'avenida de la albufera', 'calle alcala', 'calle de francisco silvela', 'calle de lopez de hoyos', 'calle embajadores', 
                         'calle serrano', 'castellana', 'paseo de la castellana, madrid', 'principe de vergara']




sum = 0

for i in range(len(df_vivienda)):
  for j in range(len(lista_calles_extensas)):
    if ((df_vivienda["street_name"][i] == lista_calles_extensas[j]) & pd.isnull(df_vivienda.street_number[i])):
      barrio = df_vivienda["barrio_nombre"][i]
      #df_vivienda["longitude"][i] = random.uniform(df_coordenadas_limites["longitud_min"][df_coordenadas_limites.nombre_barrio == barrio], df_coordenadas_limites["longitud_max"][df_coordenadas_limites.nombre_barrio == barrio])
      #df_vivienda["latitude"][i] = random.uniform(df_coordenadas_limites["latitud_min"][df_coordenadas_limites.nombre_barrio == barrio], df_coordenadas_limites["latitud_max"][df_coordenadas_limites.nombre_barrio == barrio])
      sum = sum + 1

print("Calles largas sin número recalculadas:", sum)

# Si recalculamos direcciones sin número de calle cuando la calle tiene 5 o más barrios, hay 412 direcciones recalculadas.
# Si recalculamos direcciones sin número de calle cuando la calle tiene 4 o más barrios, hay 496 direcciones recalculadas.


# Regularizamos los nombre de los barrios según el listado oficial del ayuntamiento de Madrid.



df_vivienda['barrio_nombre'].replace({'Malasaña-Universidad':'Universidad',
                                      'Conde Orgaz-Piovera':'Piovera',
                                      'Lavapiés-Embajadores':'Embajadores',
                                     'Bernabéu-Hispanoamérica':'Hispanoamérica',
                                      'Ensanche de Vallecas - La Gavia':'Ensanche de Vallecas',
                                      'El Cañaveral - Los Berrocales':'El Cañaveral',
                                      'Nuevos Ministerios-Ríos Rosas':'Rios Rosas',
                                      'San Andrés':'Villaverde Alto, Casco Histórico de Villaverde',
                                      'Cuzco-Castillejos':'Castillejos',
                                     'Sanchinarro':'Valdefuentes',
                                     'Huertas-Cortes':'Cortes',
                                     'Buena Vista':'Buenavista',
                                     'Valdebebas - Valdefuentes':'Valdefuentes',
                                     'Las Tablas':'Valverde',
                                     'Tres Olivos - Valverde':'Valverde',
                                     'Ventilla-Almenara':'Almenara',
                                     'Ambroz':'Casco Histórico de Vicálvaro',
                                     'Pau de Carabanchel':'Buenavista',
                                     'Montecarmelo':'Mirasierra',
                                     'Fuentelarreina':'Fuentelareina',
                                     'Los Cármenes':'Cármenes',
                                     'Apóstol Santiago':'Apostol Santiago',
                                     '12 de Octubre-Orcasur':'Orcasur',
                                     'Virgen del Cortijo - Manoteras':'Valdefuentes',
                                     'Valdebernardo - Valderribas':'Valdebernardo',
                                     'Arroyo del Fresno':'Mirasierra',
                                     'Campo de las Naciones-Corralejos':'Corralejos',
                                      'Palomeras sureste':'Palomeras Sureste',
                                      'Chueca-Justicia':'Justicia'}, inplace=True)


# Empezamos a descartar variables



df_vivienda_nonull = df_vivienda.drop(columns=['Column1', 'portal', 'has_private_parking', 'door', 'rent_price_by_area', 'are_pets_allowed', 'is_furnished',
        'is_kitchen_equipped', 'has_public_parking', 'sq_mt_allotment', 'is_rent_price_known', 'is_buy_price_known',
        'n_floors'])


# Como comentamos anteriormente asumimos que los valores missing de la variables dicotómicas son FALSE




df_vivienda_nonull.fillna({'is_orientation_east': False,
                    'is_orientation_south': False,
                    'is_orientation_west': False,
                    'is_floor_under': False,
                    'is_orientation_north': False,
                    'is_parking_included_in_price': False,
                    'has_public_parking': False,
                    'has_private_parking': False,
                    'has_green_zones': False,
                    'is_accessible': False,
                    'is_kitchen_equipped': False,
                    'is_furnished': False,
                    'has_storage_room': False,
                    'has_balcony': False,
                    'has_terrace': False,
                    'has_pool': False,
                    'has_garden': False,
                    'is_exterior': False,
                    'has_lift': False,
                    'has_fitted_wardrobes': False,
                    'has_ac': False,
                    'are_pets_allowed': False,
                    'has_individual_heating': False,
                    'has_central_heating': False,
                    'is_new_development': False},
                    inplace=True)


# Antes de avanzar más vamos a incluir una serie de variables con información obtenida a través del portal de datos del Ayto. de Madrid.




# Carga del CSV y creación del dataframe 
url_datos_madrid="https://raw.githubusercontent.com/ucmtfmgrupo5/database/main/datos_barrios_Madrid_valor_absoluto_csv.csv"
s=requests.get(url_datos_madrid).content
df_panel_va=pd.read_csv(io.StringIO(s.decode("UTF-8")), sep=",")

df_panel_va





df_panel_va





df_panel_va.columns





# Carga del CSV y creación del dataframe 
url_datos2_madrid="https://raw.githubusercontent.com/ucmtfmgrupo5/database/main/datos_barrios_Madrid_porcentajes_csv.csv"
s=requests.get(url_datos2_madrid).content
df_panel_p=pd.read_csv(io.StringIO(s.decode("UTF-8")), sep=",")

df_panel_p





df_panel_p.columns





df_viviendas_enriquecido = pd.merge(df_vivienda_nonull, df_panel_va,
                      how = 'left',
                      left_on ='barrio_nombre',
                      right_on = 'input.nombre_barrio',
                      indicator = True)
df_viviendas_enriquecido.drop(columns=['input.nombre_barrio', '_merge'], axis = 1, inplace = True)





df_viviendas_final = pd.merge(df_viviendas_enriquecido, df_panel_p,
                     how = 'left',
                     left_on ='barrio_nombre',
                     right_on = 'input.p_nombre_barrio',
                     indicator = True)
df_viviendas_final.drop(columns=['_merge'], axis = 1, inplace = True)





df_viviendas_final.head(5)


# Recategorización de variables




# Vemos todos los valores unicos de la columna floor para proceder a recategorizarla
df_viviendas_final['floor'].unique()





df_viviendas_final['floor']




# Recategorizamos, plantas por encima de 0 como plantas altas, las de debajo de 0 como sótanos y el resto como bajos
df_viviendas_final['floor'] = df_viviendas_final['floor'].replace(['3', '4', '1','7','6','5','2','Entreplanta exterior', '8', '9', 'Entreplanta interior',
       'Entreplanta'], 'Plantas altas')
df_viviendas_final['floor'] = df_viviendas_final['floor'].replace(['Semi-sótano exterior', 'Sótano interior',
       'Semi-sótano interior', 'Sótano', 'Sótano exterior', 'Semi-sótano'], 'Sótano')





df_viviendas_final['floor'].unique()


# Sustitución de missing en la varible built_year por la media del barrio al que pertenece la vivienda




#Exploramos la columna 'built_year'
df_viviendas_final[['built_year','barrio_nombre']]




#Calcular las medias Agrupamos por subtitle y calculamos la media
medias = df_viviendas_final[['built_year','barrio_nombre']].groupby(['barrio_nombre']).mean().round(0).astype(np.int64, errors='ignore')
medias['built_year_media']=medias['built_year']
medias




#Hacemos un merge de los dos dataframes:
df_w_medias = pd.merge(df_viviendas_final, medias, on=["barrio_nombre"])
df_w_medias.head(5)




#Actualizamos la columna original Luego del merge está se llama built_year_x, actualizamos los valores que sean NaN

df_w_medias['built_year_x'] = np.where(df_w_medias['built_year_x'].isna(),
                           df_w_medias['built_year_media'],
                           df_w_medias['built_year_x']) 

print(df_w_medias[['built_year_x', 'built_year_media']])




df_definitivo = df_w_medias




df_definitivo.drop(columns =['built_year_y', 'built_year_media', 'street_number', 'parking_price', 'sq_mt_useful', 'street_name', 'raw_address'], inplace = True)




df_definitivo.fillna({'built_year_x': 1980,
                      'sq_mt_built': df_definitivo['n_rooms']*40,
                      'n_bathrooms': df_definitivo['n_rooms'],
                      'floor': 'Plantas altas',
                      'house_type_id': 'HouseType 1: Pisos'},
                    inplace=True)

df_definitivo.built_year_x = df_definitivo.built_year_x.replace({8170 : 1970})



df_definitivo.isnull().sum().sort_values(ascending = False)




df_definitivo.dropna(inplace = True)




plt.figure(figsize=(25,15))
sns.heatmap(df_definitivo.corr(),annot=True,lw=1)




df_definitivo.columns




#Scatter plot sq_mt_built/buy_price
var = 'sq_mt_built'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
data.plot.scatter(x=var, y='buy_price', ylim=(0,9000000));




#Box plot house_type_id/buy_price
var = 'house_type_id'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);




#Box plot house_type_id/buy_price
var = 'n_rooms'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);




def func(x):
    if 0 < x <= 3:
        return '0-3'
    elif 3 < x <= 9:
        return '4-9'
    return '+9'

df_definitivo['n_rooms_cat'] = df_definitivo['n_rooms'].apply(func)




#Box plot house_type_id/buy_price
var = 'n_bathrooms'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);




#Box plot house_type_id/buy_price
var = 'has_pool'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);




#Box plot house_type_id/buy_price
var = 'has_garden'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);




# En cuanto a la variable built_year_x, no parece que haya una clara tendencia respecto al precio
var = 'built_year_x'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="buy_price", data=data)
fig.axis(ymin=0, ymax=9000000);
plt.xticks(rotation=90);




# De nuevo realizamos un análisis de correlaciones pero en este caso eligiendo variable que antes nos han parecido más relevantes

df_prueba = df_definitivo.loc[:,['buy_price', 'sq_mt_built', 'n_rooms', 'n_bathrooms', 'has_garden', 'longitude',
                                 'buy_price_by_area', 'built_year_x', 'floor', 'house_type_id', 'barrio_nombre',
                                'va_Renta neta media anual de los hogares',
                                'va_Valor catastral medio de los bienes inmuebles',
                                'va_Superficie media de la vivienda (m2) en transacción',
                                'p_Tasa absoluta paro registrado',
                                'p_ Población con estudios superiores',
                                'p_voto derecha',
                                'p_rankingvulnerabilidad']]
plt.figure(figsize=(15,8))

sns.heatmap(df_prueba.corr(),annot=True,lw=1)




#Scatterplot
sns.set()
cols = ['buy_price', 'sq_mt_built', 'n_rooms', 'n_bathrooms', 'va_Renta neta media anual de los hogares',
                                'va_Valor catastral medio de los bienes inmuebles', 'p_voto derecha']
sns.pairplot(df_definitivo[cols], size = 2.5)
plt.show();




#Histograma con curva de distribución normal (muy asimétrica y con cola muy larga, aquí sale el problema)
sns.distplot(df_definitivo['buy_price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_definitivo['buy_price'], plot=plt)




#Asimetría y curtosis
print("Skewness: %f" % df_definitivo['buy_price'].skew())
print("Kurtosis: %f" % df_definitivo['buy_price'].kurt())




#Transfomación logarítmica, de este modo intentamos que la distribución se parezca más a una normal
df_definitivo['buy_price'] = np.log(df_definitivo['buy_price'])
#df_definitivo['precio_m2_barrio'] = np.log(df_definitivo['precio_m2_barrio'])
#df_definitivo['sq_mt_built'] = np.log(df_definitivo['sq_mt_built'])
#df_definitivo['input.va_Renta neta media anual de los hogares'] = np.log(df_definitivo['input.va_Renta neta media anual de los hogares'])
#df_definitivo['input.va_Valor catastral medio de los bienes inmuebles'] = np.log(df_definitivo['input.va_Valor catastral medio de los bienes inmuebles'])
#df_definitivo['input.p_voto derecha'] = np.log(df_definitivo['input.p_voto derecha'])





#Histograma con curva de distribución normal (muy asimétrica y con cola muy larga, aquí sale el problema)
sns.distplot(df_definitivo['buy_price'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_definitivo['buy_price'], plot=plt)


# # OUTLIERS




#Outliers
sns.boxplot(df_definitivo['precio_m2_barrio'])




#Criterio del 1,5 veces el rango intercuartílico para detectar outliers

Q1=df_definitivo['precio_m2_barrio'].quantile(0.25)

Q3=df_definitivo['precio_m2_barrio'].quantile(0.75)

IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)




df_definitivo = df_definitivo[df_definitivo['precio_m2_barrio']< Upper_Whisker]
sns.boxplot(df_definitivo['precio_m2_barrio'])




df_definitivo['precio_m2_barrio'].describe()




sns.boxplot(df_definitivo['buy_price'])



#Criterio del 1,5 veces el rango intercuartílico para detectar outliers

Q1=df_definitivo['buy_price'].quantile(0.25)

Q3=df_definitivo['buy_price'].quantile(0.75)

IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)




df_definitivo = df_definitivo[df_definitivo['buy_price']< Upper_Whisker]
sns.boxplot(df_definitivo['buy_price'])




sns.boxplot(df_definitivo['sq_mt_built'])




#Criterio del 1,5 veces el rango intercuartílico para detectar outliers

Q1=df_definitivo['sq_mt_built'].quantile(0.25)

Q3=df_definitivo['sq_mt_built'].quantile(0.75)

IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3 + 1.5*IQR
print(Lower_Whisker, Upper_Whisker)




df_definitivo = df_definitivo[df_definitivo['sq_mt_built']< Upper_Whisker]
sns.boxplot(df_definitivo['sq_mt_built'])



df_definitivo




#Scatter plot sq_mt_built/buy_price
var = 'sq_mt_built'
data = pd.concat([df_definitivo['buy_price'], df_definitivo[var]], axis=1)
data.plot.scatter(x=var, y='buy_price', ylim=(8,16));




#Scatterplot
sns.set()
cols = ['buy_price', 'sq_mt_built', 'n_rooms', 'n_bathrooms', 'precio_m2_barrio']
sns.pairplot(df_definitivo[cols], size = 2.5)
plt.show();




df_definitivo.describe()




y = df_definitivo['buy_price']




df_definitivo.drop(columns =['title', 'subtitle', 'neighborhood_id', 'operation', 'rent_price', 'Direccion completa', 'barrio_id', 'distrito_id', 'buy_price', 'buy_price_by_area'], inplace = True)




df_definitivo.columns = 'input.'+ df_definitivo.columns





df_definitivo = df_definitivo.replace([True], '1')
df_definitivo = df_definitivo.replace([False], '0')




df_definitivo


# #  INPUTS DEL MODELO
# 



#Aquí elefimos las variable que queremos que tome el modelo

#df_final = df_definitivo
#df_final = df_definitivo.loc[:,['input.sq_mt_built', 'input.n_rooms','input.input.va_Valor catastral medio de los bienes inmuebles']]0.76
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms', 'input.has_garden', 'input.longitude', 'input.floor', 'input.house_type_id', 'input.barrio_nombre']]
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.has_garden', 'input.longitude','input.has_lift','input.has_pool', 'input.is_exterior', 'input.has_ac', 'input.barrio_nombre']]
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms', 'input.has_garden', 'input.longitude', 'input.buy_price_by_area', 'input.floor']]0.83
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms', 'input.has_garden', 'input.longitude', 'input.buy_price_by_area']]0.83
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms', 'input.has_garden', 'input.longitude']]0.745
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms', 'input.has_garden']]0.69
#df_final = df_definitivo.loc[:,['input.sq_mt_built','input.n_rooms', 'input.n_bathrooms']]0.7352


df_final = df_definitivo.loc[:,['input.sq_mt_built', 'input.precio_m2_barrio', 'input.house_type_id', 'input.n_rooms', 'input.is_exterior', 'input.longitude', 'input.has_lift', 'input.has_pool']]





#Convertimos las variables categóricas en dummies, si las hay
x = pd.get_dummies(df_final)
y = y




x




# Variable independiente
X = x

####(solo para variables numéricas, no categóricas)

# VIF del dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# Calculamos VIF de cada variable 
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)




# Partimos el dataset en los conjuntos de train y test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

print("X_train shape is : {}".format(X_train.shape))
print("X_test shape is : {}".format(X_test.shape))
print("y_train shape is: {}".format(y_train.shape))
print("y_test shape is: {}".format(y_test.shape))


# # Linear Regression




#Creamos el objeto de regresión lineal
model = LinearRegression()

#Entrenamos el modelo con el conjunto de train
model.fit(X_train, y_train)




model.score(X_test, y_test)




results = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
results




#Coeficiente del modelo término general
print(model.intercept_)
#Coeficientes de las variables
list(zip(x, model.coef_))




#Predicciones
y_pred_model = model.predict(X_test)  
x_pred_model = model.predict(X_train)




#Cálculo error absoluto medio
errors = abs(e**(y_pred_model) - e**(y_test))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'Euros.')




#Valor observado vs valor predicho
model_diff = pd.DataFrame({'valor_observado': e**(y_test), 'valor_predicho': e**(y_pred_model)})
model_diff.head(130)




#Plot error
plt.style.use('fivethirtyeight')
  
##Residuos parte train
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  ##Residuos parte test
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
##Línea de residuo 0
plt.hlines(y = 0, xmin = 10, xmax = 16, linewidth = 2)
  ##Leyenda
plt.legend(loc = 'upper right')
  ##Título
plt.title("Residual errors")
  
plt.show()




#En logaritmo
plt.figure(figsize=(5, 7))


ax = sns.distplot(y, hist=False, color="r", label="Valor observado")
sns.distplot(y_pred_model, hist=False, color="b", label="Valor predicho" , ax=ax)

plt.legend(loc = 'upper right')

plt.title('Valor real vs Predicción')


plt.show()
plt.close()


# # Random Forest Regressor (FINAL)





regressor = RandomForestRegressor(n_estimators = 100, random_state = 1)
regressor.fit(X_train, y_train)

regressor.score(X_test, y_test)

results = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
results


X_test


y_pred = regressor.predict(X_test)




#Cálculo error absoluto medio
errors = abs(e**(y_pred) - e**(y_test))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'Euros.')




df_eval=pd.DataFrame({'valor_observado':e**(y_test), 'valor_predicho':e**(y_pred)})
df_eval.head(130)




#Plot error
plt.style.use('fivethirtyeight')
  
##Residuos parte train
plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
  ##Residuos parte test
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
  
##Línea de residuo 0
plt.hlines(y = 0, xmin = 10, xmax = 16, linewidth = 2) 
##Leyenda
plt.legend(loc = 'upper right')
##Título
plt.title("Residual errors")
    
plt.show()




# En logaritmo
plt.figure(figsize=(5, 7))


ax = sns.distplot(y, hist=False, color="r", label="Valor observado")
sns.distplot(y_pred, hist=False, color="b", label="Valor predicho" , ax=ax)

plt.legend(loc = 'upper right')

plt.title('Valor real vs Predicción')


plt.show()
plt.close()




# Save model

currrentPath = os.path.dirname(os.path.abspath(__file__))
parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))
filename = parentPath+'/modelHousesModel/modelHousesModel'

outfile = open(filename,'wb')

pickle.dump(model, outfile)

outfile.close()






