import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#url = "output.csv"
#"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv("../Data/output.csv")
print(df)

features = ['score1', 'score2', 'score3', 'score4', 'score5', 'score6']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
#y = df.loc[:,['target']].values
#Standardizing the features
x = StandardScaler().fit_transform(x)

print(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
x = StandardScaler().fit_transform(principalComponents)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf)
#principalDf.to_csv(path_or_buf= 'output2.csv', columns= ['principal component 1',  'principal component 1'])
principalDf = principalDf.values
print(principalDf)

