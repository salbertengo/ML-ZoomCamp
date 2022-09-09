from socketserver import DatagramRequestHandler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




data = pd.read_csv('data.csv')

df = data.loc[data['Make'] == 'Lotus']
df.drop_duplicates()
df3 = pd.concat([df["Engine HP"], df["Engine Cylinders"]])
print(df3.head())

X = df3.to_numpy()
XT = X.transpose()
XTX = np.matmul(X,XT)
IXTX = np.linalg.inv(XTX)
y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])