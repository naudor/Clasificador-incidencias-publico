#!/usr/bin/env python
# coding: utf-8

# # Llegim les dades

# In[1]:


import pandas as pd
import csv

print("Leyendo datos de entrenamiento");

df = pd.read_csv("DataSets/TOTES_LES_INCIDENCIES_v4.csv",  encoding = "ISO-8859-1")


# # Limpaimos las filas que no estan completas

# In[2]:


df = df.dropna(subset = ["caller_id","short_description","description","assignment_group"])


# # Como solo podemos comparar dos campos pero quiero utilizar 3, junto los dos campos de datos a valorar
# In[3]:


df['concatenat'] = df['caller_id'] +'.\r\n'+ df['short_description']


# # Factoritzemos las categorias
# In[5]:


df['category_id'] = df['assignment_group'].factorize()[0]


# # Función para quitar los acentos
# In[5]:


def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s


# # Limpiamos el texto y vectorizamos las descripciones
# In[6]:


print("Vectorizando descripciones")

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.es.stop_words import STOP_WORDS as es_stop
from io import StringIO
import string

df['concatenat'] = df['concatenat'].apply(lambda fila: fila.lower())
df['short_description'] = df['short_description'].apply(lambda fila: normalize(fila))

final_stopwords_list = list(es_stop)
final_stopwords_list.append('\r\n')
final_stopwords_list.append(string.punctuation)
final_stopwords_list.append("mg")
final_stopwords_list.append("comp")
final_stopwords_list.append("kp")

# convierte el texto en vectores con la frecuencia de las palabras
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=final_stopwords_list,max_features=15000)
features = tfidf.fit_transform(df.concatenat).toarray()
labels = df.category_id

category_id_df = df[['assignment_group', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'assignment_group']].values)
#features.shape


# # Entrenamos el modelo
# In[7]:


print("Entrenando modelo")

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[8]:


from sklearn import metrics

unic_label_train = df.groupby(['assignment_group'])['assignment_group'].size()
unic_label_train = unic_label_train[unic_label_train > 4].index.get_level_values(0).tolist()

#print(metrics.classification_report(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred, 
                                  target_names=unic_label_train))


# # Guardamos el modelo
# In[9]:


print("Guardando modelo")

import pickle
from tempfile import TemporaryFile

with open("clasificacion_CAUS_modelo.pickle", "wb") as file:
    pickle.dump(model, file)

with open("clasificacion_CAUS_vectorizador.pickle", "wb") as file:
    pickle.dump(tfidf, file)

with open("clasificacion_CAUS_categorias.pickle", "wb") as file:
    pickle.dump(id_to_category, file)
    
print ("Modelo guardado")


# In[ ]:




