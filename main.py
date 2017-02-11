import pandas as pandas
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pickle

#loading dataframe
df = quandl.get("YAHOO/GOOGL")

#Collecting columns that are important
df = df[['Open','High','Low','Adjusted Close','Volume']]

#Features
df['HL_PCT'] = (df['High'] - df['Low']) / df['Adjusted Close'] * 100.0
df['PCT_change'] = (df['Adjusted Close'] - df['Open']) / df['Open'] * 100.0

#Creating new data frame with features
df = df[['Adjusted Close','HL_PCT','PCT_change','Volume']]

forecast_col = 'Adjusted Close'
df.fillna(-9999999, inplace=True)

#Rounding up to a whole number
forecast_out = int(math.ceil(0.01*len(df)))

#Defining label
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Defining Classifier
# clf = LinearRegression()
# clf.fit(X_train, y_train)

# with open('linearregression.pickle', 'wb') as f:
# 	pickle.dump(clf,f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately);

print(forecast_set, accuracy, forecast_out)


