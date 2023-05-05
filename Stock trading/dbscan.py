#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:52:23 2023

@author: dixondomfeh
"""


import numpy as np
import pandas as pd
from IPython import get_ipython

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-darkgrid')

# Pairs
from itertools import combinations


#%% Feature engineering

path = '/Users/dicksonnkwantabisa/Desktop/CS7643_Deep_Learning/Project/Unsupervised-Learning-Resources/data_modules/'
dat = pd.read_csv(path + 'SP_500_data.csv', index_col=0)

returns_data = dat.pct_change().dropna()
print(f'shape of data {returns_data.shape}')

# dimensionality reduction
desired_variance = 0.90

pca = PCA(n_components=desired_variance)
pca.fit(returns_data)

# extract principal components
X = pca.components_
print(f'Number of features in reduce dimension: {X.T.shape[1]}')

# add these components to fundamental data of each stock
dat = pd.read_csv(path + 'fundamentals.csv', index_col=0)

for col in dat.columns:
    X = np.vstack((X, dat[col].values))
    
print(f'Shape  of final dataset: {X.T.shape}')

# convert data into a dataframe
X = pd.DataFrame(X.T, index=returns_data.columns)
X.columns = ['Feature_' + str(i) for i in range(1,82)]
X.head()


# standardize data
np.round(X.describe(), 2)

X = preprocessing.StandardScaler().fit_transform(X,)
X = pd.DataFrame(X, index=returns_data.columns)
X.columns = ['Feature_' + str(i) for i in range(1,82)]
X.head()

pd.DataFrame(X).to_csv(path + "SP500_principal_components.csv")


#%% Create Pairs using DBSCAN

X = pd.read_csv(path + 'SP500_principal_components.csv', index_col=0)

#get tickers
tickers = X.index

dbscan = DBSCAN(eps=5, min_samples=3)
dbscan.fit(X)

'''
If you're getting an error, upgrade 

pip install threadpoolctl==3.1.0

'''
# get labels of clusters
labels = dbscan.labels_



'''
The number of clusters created will be equal to the number of unique labels. 
Note that, -1 will be assigned to the noisy points. Hence, 
I subtract from the unique count if -1 is present in the labels.
'''

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print(f'\nCusters discovered: {n_clusters_}')


# create a series with cluster labels
clustered_series_all = pd.Series(index=tickers, data=labels.flatten())

# remove data points with -1
clustered_series = clustered_series_all[clustered_series_all != -1]


#%% visualize the 81 dimensional space in 2D 
X_tsne = TSNE(n_components=2,
              learning_rate=100,
              perplexity=25,
              random_state=42).fit_transform(X)

# plot clusters
plt.figure(1, facecolor='white', figsize=(15,15))
plt.axis('off')
plt.scatter(
    X_tsne[(labels!=-1), 0],
    X_tsne[(labels!=-1), 1],
    s=200,
    alpha=0.8,
    c=labels[labels!=-1],
    cmap='viridis'
    )
# plot the noise
plt.scatter(
    X_tsne[(clustered_series_all==-1).values, 0],
    X_tsne[(clustered_series_all==-1).values, 1],
    s=200,
    alpha=0.05
    )
plt.title('Cluusters', fontsize=14)
plt.show()

# check the number of stocks in each cluster
clustered_series.value_counts().plot.bar(figsize=(15,5))
plt.title("Cluster Wise Nuumber of Stocks", fontsize=14)
plt.xlabel("Cluster Label", fontsize=12)
plt.ylabel("Number of Stocks", fontsize=12)
plt.show()


#%% Creat pairs

clustered_series[clustered_series == 8]

"""
Interestingly, cluster 0 is a list of enrgy utility companies
cluster 5 is banks

you can check the rest!
"""

stocks_pair = {}

for i in range(0, n_clusters_):
    pairs = list(combinations(clustered_series[clustered_series == i].index, 2))
    stocks_pair[i] = pairs


stocks_pair = pd.DataFrame.from_dict(stocks_pair.items())
stocks_pair.rename(columns={0:'Cluster No.', 1: 'Pairs'}, inplace=True)
stocks_pair = stocks_pair.set_index('Cluster No.')

# number of pairs in each cluster
stocks_pair['Number of Pairs'] = stocks_pair['Pairs'].apply(lambda x:len(x))

'''
I will leave it to the group to decide on the pairs to trade
'''

stocks_pair.iloc[5]['Pairs']

#%% Test for cointegration

'''
Two stocks might not be stationary individually. But we can create a portfolio 
of these two stocks such that the portfolio is stationary. In such cases, 
we call the stocks to be cointegrated. 

'''

pairs = []
for cluster in stocks_pair.index:
    group = pd.DataFrame(stocks_pair.Pairs[cluster])
    pairs.append(group)
     
stock_pairs = pd.concat([pairs[0],pairs[1],pairs[2],pairs[3],pairs[4],pairs[5],pairs[6],pairs[7],pairs[8]],axis=0)
stock_pairs.rename(columns={0:'Stock1', 1:'Stock2'}, inplace=True)

dat = pd.read_csv(path + 'SP_500_data.csv', index_col=0)

def is_coint(pair):
    """Function to check for cointegration
    """
    
    # Data for the two stocks in pair 
    data = pd.DataFrame()
    data['stock_1'] = dat[pair[0]]
    data['stock_2'] = dat[pair[1]]
    data = data.dropna()

    # Create a portfolio using a hedge ratio obtained using linear regression
    model = sm.OLS(data['stock_1'], data['stock_2'])
    model = model.fit() 
    portfolio = (data['stock_1'] - model.params[0] * data['stock_2'])

    # Calculate the test statistics and print the results at 90% level of confidence
    result = ts.adfuller(portfolio)
    
    if result[0] < result[4]['10%']:
        return True
    else:
        return False
    
stock_pairs['cointegrated'] = stock_pairs.apply(is_coint, axis=1)


print(f'There are {len(stock_pairs[stock_pairs.cointegrated == True])}\n stock pairs that are cointegrated at the 90% CI!')

cointegrated_stocks = stock_pairs[stock_pairs.cointegrated == True]
cointegrated_stocks.to_csv(path+'cointegrated_stocks_SP500.csv', index=False)


























