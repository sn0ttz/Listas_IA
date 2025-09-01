#!/usr/bin/env python
# coding: utf-8

# **Atualizando a biblioteca para plotagem de gráficos**
# 

# In[1]:




# **Importando bibliotecas**

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# **Abrindo um arquivo CSV do drive**

# In[3]:


base = pd.read_csv('./Lista03/titanic/train.csv')

# **Você também pode carregar seu arquivo e já selecionar as colunas que desejar... investigue esta função**

# In[4]:


#base2 = pd.read_csv('/content/sample_data/restaurante_correto.csv', ';', usecols=['Alternativo', 'Bar'])
#base2

# In[5]:


base

# In[6]:


base.head(3)

# In[7]:


base.tail(2)

# **Contando quantidade de instâncias**
# 

# In[8]:


Classificação = 'Survived'
print(f"Distribuição de {Classificação}:")
print(base[Classificação].value_counts())

# In[9]:


sns.countplot(x = base[Classificação]);

# **Tratamento de dados categóricos**

# > *LabelEncoder - Vamos tratar os dados categóricos colocando 1, 2, 3 e etc**
# 
# 

# In[10]:


from sklearn.preprocessing import LabelEncoder

# In[11]:


#para codificar todos os atributos para laberEncoder de uma única vez
#base_encoded = base.apply(LabelEncoder().fit_transform)

# Tratamento de dados faltantes
base['Age'].fillna(base['Age'].median(), inplace=True)
base['Embarked'].fillna(base['Embarked'].mode()[0], inplace=True)

# Remover colunas não numéricas que não são úteis para o modelo
base = base.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Codificação de atributos categóricos
cols_label_encode = ['Sex', 'Embarked']
base[cols_label_encode] = base[cols_label_encode].apply(LabelEncoder().fit_transform)

# In[12]:


base

# 
# 
# >** OneHotEncoder - Agora vamos binarizar atributos não ordinais**

# **Contando quantas opções de resposta tem cada atributo**

# In[13]:


print(f"Valores únicos em Sex: {base['Sex'].unique()}")
print(f"Valores únicos em Pclass: {base['Pclass'].unique()}")
print(f"Valores únicos em Embarked: {base['Embarked'].unique()}")

# In[14]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# In[15]:


# Não usar OneHotEncoder, manter todas as colunas como estão
base_encoded = base.copy()

# In[16]:


base_encoded

# In[17]:


base_encoded.shape

# **Separar o dataset em variáveis independentes (X_prev) e dependentes (y_classe)**

# In[18]:


# Separar features (X) e target (y)
X_prev = base_encoded.drop(['Survived'], axis=1)
y_classe = base_encoded['Survived']

# **Método de amostragem Holdout**

# In[19]:


from sklearn.model_selection import train_test_split

# In[20]:


X_prev

# In[21]:


y_classe

# In[22]:


y_classe.shape

# In[23]:


#X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y)
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 42)

# In[24]:


X_treino.shape

# In[25]:


X_teste.shape

# In[26]:


X_teste

# In[27]:


y_treino

# In[28]:


y_teste

# In[29]:


import pickle

# In[30]:


with open('./Titanic.pkl', mode = 'wb') as f:
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)
