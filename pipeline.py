from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Decision Tree:-> ", accuracy_score(Y_test, predictions))



from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Kneighbors classifier:-> ",accuracy_score(Y_test, predictions))



from sklearn.ensemble import RandomForestClassifier
my_classifier =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Random Forest Classifier:-> ",accuracy_score(Y_test, predictions))




from sklearn.ensemble import AdaBoostClassifier
my_classifier =  AdaBoostClassifier()

my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Ada Boost Classifier:-> ",accuracy_score(Y_test, predictions))




from sklearn.neural_network import MLPClassifier
my_classifier =  MLPClassifier(alpha=1)

my_classifier.fit(X_train, Y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("MLP Classifier:-> ",accuracy_score(Y_test, predictions))