import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor


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

df_v2.info()
df_v2.describe()

df_v2.reset_index(drop=True,inplace = True)
X_train, X_test, y_train, y_test = train_test_split(df_v2.iloc[:, :-1] , df_v2.iloc[:, -1], test_size=0.2)

categorical_features = [0, 1]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeRegressor(max_depth=50))]
)

clf.fit(X_train.values, y_train.values)
print("model score: %.3f" % clf.score(X_test.values, y_test.values))
