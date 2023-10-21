import pandas as pd
import sklearn
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

total_data = pd.read_csv("Seed_Data.csv")

#print(total_data.head())

data = total_data.iloc[:, 0:7]

#print(data.head())

y = total_data.iloc[:, 7]
#y = total_data.drop(["A", "P", "C", "LK", "WK", "A_Coef", "LKG"], axis = 1)

#print(y.head())

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size = .1)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

clf = svm.SVC()
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

print(accuracy_score(prediction, y_test))