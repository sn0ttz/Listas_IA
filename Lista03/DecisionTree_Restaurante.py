#!/usr/bin/env python
# coding: utf-8

# **Vamos experimentar agora o algoritmo Decision Tree?**

# In[57]:

# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# In[59]:


import pickle
with open('./Lista03/Titanic.pkl', 'rb') as f:
  X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# In[60]:


modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

# 
# 
# > **Vamos testar o modelo?**
# 
# 

# In[61]:


previsoes = modelo.predict(X_teste)

# In[62]:


previsoes

# 
# 
# > **Será se o modelo acertou?**
# 
# 

# In[63]:


y_teste

# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_teste,previsoes)

# In[65]:


from yellowbrick.classifier import ConfusionMatrix
confusion_matrix(y_teste, previsoes)

# In[66]:


cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

# In[67]:
# Adicione estas linhas antes da linha 89

print(classification_report(y_teste, previsoes))

# In[68]:


from sklearn import tree
previsores = X_treino.columns.tolist()  # Converter para lista
print(f"Classes do modelo: {modelo.classes_}")
print(f"Número de classes: {len(modelo.classes_)}")

figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
tree.plot_tree(modelo, feature_names=previsores, class_names=['Morreu', 'Sobreviveu'], filled=True, fontsize=10, rounded=True)
plt.title("Árvore de Decisão - Titanic (Padrões de Sobrevivência)", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Análise das regras de decisão
from sklearn.tree import export_text
print("\n=== REGRAS DE DECISÃO - PADRÕES DE SOBREVIVÊNCIA ===")
tree_rules = export_text(modelo, feature_names=previsores, max_depth=4)
print(tree_rules)

print("\n=== IMPORTÂNCIA DAS FEATURES ===")
for feature, importance in zip(previsores, modelo.feature_importances_):
    if importance > 0.01:
        print(f"{feature}: {importance:.3f}")
