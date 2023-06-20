import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import numpy
from sklearn.svm import LinearSVC, SVC

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

data = pandas.read_csv("ukol_04_data.csv")

y = data["y"]

categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "campaign", "poutcome"]
numeric_columns = ["age", "balance", "duration", "pdays", "previous"]
numeric_data = data[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Rozhodovací strom a jeho omezení na 4 patra
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["no", "yes"])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('tree.png')

# Vytvoření matice záměn:
#ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
#plt.show()

# Určení hodnoty metriky accuracy:
scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#print(accuracy_score(y_test, y_pred))
# accuracy_score = 0.9018096514745308, tedy 90,2%

# Pro marketingové oddělení vyberu metriku precision score - penalizuji, pokud dojde k označení klienta, který zájem nemá, za klienta, který zájem má
#print(precision_score(y_test, y_pred, pos_label="yes"))
# precision_score = 0.6481927710843374, tedy 64,82%

# Využij algoritmus K Nearest Neighbours k predikci, zda si klient/klientka založí termínovaný účet. Pomocí cyklu (nebo pomocí GridSearchCV) urči počet uvažovaných sousedů, které algoritmus bere v úvahu. Uvažuj následující hodnoty parametru: 3, 7, 11, 15, 19, 23. Jaká je nejlepší hodnota metriky? 
y = data["y"]

categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "campaign", "poutcome"]
numeric_columns = ["age", "balance", "duration", "pdays", "previous"]
numeric_data = data[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

clf1 = KNeighborsClassifier()
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

#ks = range(1, 24, 2)
#precision_scores = []

#for k in ks:
#    clf1 = KNeighborsClassifier(n_neighbors=k)
#    clf1.fit(X_train, y_train)
#    y_pred1 = clf1.predict(X_test)
#    precision_scores.append(precision_score(y_test, y_pred, pos_label="yes"))
#plt.plot(ks, precision_scores)
#plt.show()

#clf2 = KNeighborsClassifier(n_neighbors=23)
#clf2.fit(X_train, y_train)
#y_pred2 = clf2.predict(X_test)
#print(precision_score(y_test, y_pred2, pos_label="yes"))

# hodnota precision score pro parametr 3 = 45,43%
# hodnota precision score pro parametr 7 = 49,58%
# hodnota precision score pro parametr 11 = 53,14%
# hodnota precision score pro parametr 15 = 54,69%
# hodnota precision score pro parametr 19 = 54,99% ==> nejvyšší, ale je nižší než u rozhodovacího stromu
# hodnota precision score pro parametr 23 = 54,94%


# Jako druhý využij algoritmus Support Vector Machine. Využij lineární verzi, tj. LinearSVC.
y = data["y"]

categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "campaign", "poutcome"]
numeric_columns = ["age", "balance", "duration", "pdays", "previous"]
numeric_data = data[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf3 = LinearSVC()
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
print(precision_score(y_test, y_pred3, pos_label="yes"))
# precision_score = 0.6437823834196891, tedy 64,38%, ale pokaždé vyjde trochu jiná hodnota