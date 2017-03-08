from sklearn import datasets, preprocessing, neighbors, model_selection
from numpy import linspace

data = datasets.load_boston()
sdata = preprocessing.scale(data.get('data'))
for i in linspace(start=1, stop=10, num=200):
    neigh = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
    print model_selection.cross_val_score(neigh.fit, scoring='neg_mean_squared_error')

