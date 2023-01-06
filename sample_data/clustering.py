from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


df = pd.read_csv("D:\Thesis\\acoustics\Data collected\energy.csv")
column = df.head()
print(df.energy)
plt.scatter(df.energy,df.energy_noise)
plt.show()

km = KMeans(n_clusters=2)
y_predict = km.fit_predict(df[['energy','energy_noise']])

df.cluster = y_predict

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]

plt.scatter(df1.energy,df1.energy_noise,color='green')
plt.scatter(df2.energy,df2.energy_noise,color='red')


plt.show()