import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def sale_price_replace(house_prices):
    df = house_prices.copy()
    cluster_calc = df[['GrLivArea', 'SalePrice','1stFlrSF','GarageArea']]
    cluster_calc.dropna()
    cluster_set = breif_clustering(cluster_calc, 3)
    
    tem = df[df.columns]
    tem['SpThird'] = cluster_set['Cluster']
    tem.pop('SalePrice')
    return tem


def breif_clustering(X, n_clusters):

    X_pca = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)

    X['Cluster'] = km.fit_predict(X_pca)
    centroides = km.cluster_centers_
    return X