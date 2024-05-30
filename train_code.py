import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import pickle


url_data = "https://raw.githubusercontent.com/danamz31/DPS-DanaAlzoubi/main/monatszahlen2402_verkehrsunfaelle_export_29.csv"
df = pd.read_csv(url_data)

df = df[['MONATSZAHL','AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']]

df_v1 = df[df['JAHR']<2021]

df_v2 = df_v1[df_v1['MONAT'] != 'Summe']

df_v2['MONAT'] = df_v2['MONAT'].apply(lambda x:x[-2:])

df_v2.groupby('MONATSZAHL')['WERT'].sum()
plt.figure(figsize=(8,6))
grouped_data = df_v2.groupby('MONATSZAHL')
summation = grouped_data['WERT'].sum()
plt.bar(summation.index, summation.values)
plt.xlabel('MONATSZAHL')
plt.ylabel('WERT')
plt.title('Number of Accidents per Category')
plt.show()
