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


