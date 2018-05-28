import pandas as pd
import sklearn as sk

data = pd.read_csv("iris.data", names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Type"])
print(data)
