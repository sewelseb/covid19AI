import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import joblib
from sklearn.metrics import mean_absolute_error


def remove_useless_collumns(data):
    del data['day']
    del data['month']
    del data['year']
    del data['geoId']
    del data['countryterritoryCode']
    # del data['deaths']
    return data

pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("covidData.csv", sep=";")

#remove useless collumn
data = remove_useless_collumns(data)

data = data.dropna()



#Change date to number of days since 1jan
data['day'] = pd.to_datetime(data['dateRep']) - pd.to_datetime("31-12-2019")
data['day'] = data['day'].dt.days
del data['dateRep']

# Replace categorical data with one-hot encoded data
features_data = pd.get_dummies(data, columns=['countriesAndTerritories'])
features_data.dropna()

del features_data['cases']



X = features_data.to_numpy() # dans l'exemple du cours, on utilise as_matrix, mais il a été remplacé en as_numpy
y = data['cases'].to_numpy()

print(list(features_data.columns))

# split the data between test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=7,
    max_features=0.1,
    loss='huber'
)

# create the model based on experiacne
model.fit(X_train, y_train)
joblib.dump(model, 'trained_country_new_case_predictor.pkl')


# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)

print("Score: %.4f" % int(model.score(X_test, y_test)*100))