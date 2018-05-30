import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

'''The following is the training of a logistic regression model upon the iris dataset. Only 70% of the data
is used and there are no predictions made within this file.'''


data = pd.read_csv("iris.data", names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Type"])

data = data.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(data, train_size=.7, test_size=.3)

model = LogisticRegression()

train_X, train_y = np.split(train,[-1],axis=1)

model.fit(train_X, train_y)

test_X, test_y = np.split(test, [-1], axis=1)