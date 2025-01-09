# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:34:47 2024

@author: 34645
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Cargar el dataset original
df = pd.read_csv('Student Depression Dataset Clean.csv')

# Aplicar preprocesamiento (asegúrate de incluir tus ajustes anteriores)
df = df.drop(columns=['id', 'Work Pressure'])  # Eliminar columnas redundantes o irrelevantes

from sklearn.model_selection import train_test_split

# Asumiendo 'class' es una columna categórica clave
df_sample, _ = train_test_split(df, train_size=4166/27830, stratify=df['Depression'], random_state=42)


# Normalización de los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
d_std = scaler.fit_transform(df_sample.drop(columns=['Degree']))

# K-Means - Aplicar clustering con el número óptimo de clústeres
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(d_std)

# Agregar los clústeres al DataFrame
df_sample['Cluster'] = clusters

# Mostrar características promedio de cada clúster
print("Medias de variables por clúster:")
cluster_summary = df_sample.groupby('Cluster').mean()
print(cluster_summary)

# Visualización con t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
d_tsne = tsne.fit_transform(d_std)


# Crear un DataFrame con las coordenadas t-SNE
tsne_df = pd.DataFrame(d_tsne, columns=['X', 'Y'])

# Combinar la tabla original con las coordenadas t-SNE y clústeres
final_df = pd.concat([df_sample.reset_index(drop=True), tsne_df], axis=1)

# Guardar el DataFrame combinado en un archivo CSV
final_df.to_csv('tsne_clusters_completo.csv', index=False)
print("Archivo 'tsne_clusters_completo.csv' creado con éxito.")


# Graficar t-SNE con los clústeres de K-Means
plt.figure(figsize=(8, 6))
plt.scatter(d_tsne[:, 0], d_tsne[:, 1], c=clusters, cmap='viridis', s=5, alpha=0.7)
plt.title("t-SNE con clústeres de K-Means")
plt.xlabel("Dimensión 1")
plt.ylabel("Dimensión 2")
plt.colorbar(label='Cluster')
plt.show()

"""
# Método del Codo para determinar el número óptimo de clústeres
distortions = []
for i in range(1, 11):  # Probar entre 1 y 10 clústeres
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(d_std)
    distortions.append(km.inertia_)

# Graficar la inercia (distorsión) vs número de clústeres
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), distortions, marker='o')
plt.title("Método del codo para elegir el número de clústeres")
plt.xlabel("Número de Clústeres")
plt.ylabel("Inercia")
plt.show()
"""

# Combinar la tabla original (sin ID) con las coordenadas t-SNE y clústeres
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular medias de cada variable por clúster
cluster_summary = df_sample.groupby('Cluster').mean()

# Crear un heatmap para visualizar las diferencias
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_summary.T, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Medias de variables por Clúster")
plt.show()