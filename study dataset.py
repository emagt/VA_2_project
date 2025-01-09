# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:25:35 2024

@author: 34645
"""

import timeit
import pandas as pd
import numpy as np
import seaborn as sns # for visualization
import matplotlib.pyplot as plt
import plotly.express as px


# Ruta del archivo Excel
ruta_archivo_excel = 'Student Depression Dataset.csv'
df = pd.read_csv(ruta_archivo_excel)
print(df.head())

for column in df.columns:
    print(f"Valores únicos en la columna '{column}':")
    print(df[column].unique())
    print("-" * 50)  # Separador entre columnas

print(df.dtypes)

####Limpieza de los datos##########

df = df.dropna(subset=['Financial Stress'])
# Convertir directamente los valores de las columnas a minúsculas para eliminar correctamente las filas
data_cleaned = df[
    ~((df['Sleep Duration'].str.lower() == 'others') |
      (df['Dietary Habits'].str.lower() == 'others'))
]


# Eliminar filas donde la columna 'City' tenga valores específicos: 'city', '3.0', 'less than 5.Kaylan'
values_to_remove = ['city', '3.0', 'less than 5.kaylan']
data_cleaned = data_cleaned[~data_cleaned['City'].str.lower().isin(values_to_remove)]

#cambiar los valores a horas de sleep duration
sleep_duration_mapping = {
    '5-6 hours': 6,
    'Less than 5 hours': 5,
    '7-8 hours': 7,
    'More than 8 hours': 8
}

# Aplicar la transformación a la columna 'Sleep Duration'
data_cleaned['Sleep Duration'] = data_cleaned['Sleep Duration'].map(sleep_duration_mapping)

#Convertir las columnas float a int
columns_to_convert = ['Age', 'Academic Pressure','Sleep Duration', 'Work Pressure','Study Satisfaction','Job Satisfaction','Work/Study Hours','Financial Stress']
for col in columns_to_convert:
    # Asegurarse de que no haya valores NaN antes de convertir
    if data_cleaned[col].isnull().sum() == 0:
        data_cleaned[col] = data_cleaned[col].astype('int64')

#print(data_cleaned.dtypes)



#Convertir texto categórico a tipo categórico
categorical_columns = [
    'Gender',
    'City',
    'Profession',
    'Dietary Habits',
    'Degree',
    'Have you ever had suicidal thoughts ?',
    'Family History of Mental Illness'
]
for col in categorical_columns:
    data_cleaned[col] = data_cleaned[col].astype('category')

# Eliminar la columna City
data_cleaned = data_cleaned.drop(columns=['City'])
# Convertir Profession a binaria (Student = 1, otros = 0)
data_cleaned['Profession'] = data_cleaned['Profession'].apply(lambda x: 1 if x == 'Student' else 0)


#quiero convertir a 0 y 1 diferentes categorias y que sean numericas
gender_mapping = {'Male': 0, 'Female': 1}
data_cleaned['Gender'] = data_cleaned['Gender'].map(gender_mapping)

result_mapping = {'No': 0, 'Yes': 1}
data_cleaned['Have you ever had suicidal thoughts ?'] = data_cleaned['Have you ever had suicidal thoughts ?'].map(result_mapping)
data_cleaned['Family History of Mental Illness'] = data_cleaned['Family History of Mental Illness'].map(result_mapping)


# Convertir Dietary Habits
dietary_mapping = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
data_cleaned['Dietary Habits'] = data_cleaned['Dietary Habits'].map(dietary_mapping)


degree_mapping = {
    'Class 12': 0,  # Educación Secundaria
    'B.Pharm': 1, 'BSc': 1, 'BA': 1, 'BCA': 1, 'B.Ed': 1, 'LLB': 1, 'BE': 1,
    'BHM': 1, 'B.Com': 1, 'B.Arch': 1, 'B.Tech': 1, 'BBA': 1,  # Grado Universitario
    'M.Tech': 2, 'M.Ed': 2, 'MSc': 2, 'M.Pharm': 2, 'MCA': 2, 'MA': 2,'LLM':2,
    'MBA': 2, 'M.Com': 2, 'ME': 2, 'MHM': 2, 'MD': 2, 'MBBS': 2,  # Postgrado/Máster
    'PhD': 3  # Doctorado
}

# Aplicar el mapeo al dataset
data_cleaned['Degree'] = data_cleaned['Degree'].map(degree_mapping)
data_cleaned = data_cleaned.dropna(subset=['Degree'])






columns_to_convert = ['Degree','Gender', 'Have you ever had suicidal thoughts ?',
                      'Family History of Mental Illness','Dietary Habits','Profession']
for col in columns_to_convert:
    # Asegurarse de que no haya valores NaN antes de convertir
    if data_cleaned[col].isnull().sum() == 0:
        data_cleaned[col] = data_cleaned[col].astype('int64')



#eliminar valores extremos
data=data_cleaned
numeric_columns = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress', 'Academic Pressure', 'Work Pressure']
for col in numeric_columns:
    # Opcional: Si hay valores fuera de rangos típicos, podemos removerlos
    data = data[data[col].between(data[col].quantile(0.01), data[col].quantile(0.99))]



# Guardar el archivo actualizado como Excel y CSV
data_cleaned.to_csv('Student Depression Dataset Clean.csv', index=False)


print("Archivo actualizado y guardado como 'Student Depression Dataset Clean.csv'")