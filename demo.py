import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble
import joblib
from sklearn.metrics import mean_absolute_error
import sklearn


def remove_useless_collumns(data):
    del data['day']
    del data['month']
    del data['year']
    del data['geoId']
    del data['countryterritoryCode']
    del data['deaths']
    return data

pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("covidData.csv", sep=";")
data = data.dropna()

#remove useless collumn
data = remove_useless_collumns(data)



#Change date to number of days since 1jan
data['day'] = pd.to_datetime(data['dateRep']) - pd.to_datetime("31-12-2019")
data['day'] = data['day'].dt.days
del data['dateRep']


# Replace categorical data with one-hot encoded data
features_data = pd.get_dummies(data, columns=['countriesAndTerritories'])
features_data = sklearn.utils.shuffle(features_data)

y = features_data['cases']
del features_data['cases']

X = features_data.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ensemble.GradientBoostingRegressor(
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth = 6,
    min_samples_leaf = 7,
    max_features = 0.1,
    loss = 'huber'
)

model.fit(X_train, y_train)

joblib.dump(model, 'trained_ai.pkl')

mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

print("Score: %.4f" % int(model.score(X_test, y_test)*100))

