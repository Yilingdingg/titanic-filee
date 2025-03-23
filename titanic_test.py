import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv("titanic.csv")

x = data.iloc[:, 0:8].values
y = data.iloc[:, -1].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(Ct.fit_transform(x))
print(x)

Le = LabelEncoder()
y = Le.fit_transform(y)
print(y)
name = data["Name"]
data["Name"] = Le.fit_transform(name)
print(data)
Sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train[0:,1:] = Sc.fit_transform(x_train[0:,:])
x_test[0:,1:] = Sc.transform(x_test[0:,1:])
print(x_train)
