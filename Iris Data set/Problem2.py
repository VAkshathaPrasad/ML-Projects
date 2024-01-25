from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X = iris.data[iris.target < 2]
y = iris.target[iris.target < 2]
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
 test_size = 0.2, random_state = 15412372)
LR_Classifier = LogisticRegression()
LR_Classifier.fit(X_train, y_train)
coefficients = LR_Classifier.coef_
print("Coefficients:")
print(coefficients)
y_predicted = LR_Classifier.predict(X_test)
y_train_predicted = LR_Classifier.predict(X_train)
print( classification_report(y_test, y_predicted) )
print( accuracy_score(y_test, y_predicted))


#For all the classes
X1 = iris.data
y1 = iris.target
X1_scaled = StandardScaler().fit_transform(X1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1_scaled, y1, 
 test_size = 0.2, random_state = 15412372)
LR_Classifier = LogisticRegression()
LR_Classifier.fit(X1_train, y1_train)
coefficients1 = LR_Classifier.coef_
print("Coefficients:")
print(coefficients1)
y1_predicted = LR_Classifier.predict(X1_test)
y1_train_predicted = LR_Classifier.predict(X1_train)
print( classification_report(y1_test, y1_predicted) )
print( accuracy_score(y1_test, y1_predicted))