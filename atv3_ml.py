#%%
from turtle import title
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from xarray import corr
# %%
df = pd.read_excel('./barrettII_eyes_clustering.xlsx')
#%%
f1 = df['Correto'] == 'S'
f2 = df['Correto'] == 'N'
df.loc[f1, ['Correto']] = 1
df.loc[f2, ['Correto']] = 0
# %%
columns = ['AL', 'ACD', 'WTW', 'K1', 'K2']
# %%
X = df[columns]
# %%
grafico = px.scatter_matrix(df, dimensions = columns, color = 'Correto')
grafico.show()
#%%
cor = df.corr()
sns.heatmap(cor, annot=True)
plt.show()
# %%
model1 = KMeans(init='k-means++', random_state=11, n_clusters=4)
from yellowbrick.cluster.elbow import kelbow_visualizer
visualizer = kelbow_visualizer(KMeans(random_state=11), X, k=(1,11))
# %%
model1.fit(X)
#%%
centroids = model1.cluster_centers_
labels = model1.labels_
#%%
Y = X
Y.loc[:,'label'] = labels
Y
#%%
filtro1 = Y['label'] == 0
filtro2 = Y['label'] == 1
filtro3 = Y['label'] == 2
filtro4 = Y['label'] == 3
# %%
grupo1 = Y[filtro1].drop(columns=['label'])
grupo2 = Y[filtro2].drop(columns=['label'])
grupo3 = Y[filtro3].drop(columns=['label'])
grupo4 = Y[filtro4].drop(columns=['label'])
# %%
grupo1.describe()
# %%
grupo2.describe()
# %%
grupo3.describe()
# %%
grupo4.describe()
# %%
grupo1.boxplot()
# %%
grupo2.boxplot()
# %%
grupo3.boxplot()
# %%
grupo4.boxplot()
# %%
centroids.round(3)
# %%
Y