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
import numpy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

data = pandas.read_csv("bodyPerformance.csv")

y = data["class"]

categorical_columns = ["gender"]
numeric_columns = ["age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce"]
numeric_data = data[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# rozhodovací strom a jeho omezení
clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["A", "B", "C", "D"])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('tree.png')

# Vytvoření matice záměn:
#ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
#plt.show()
# do skupiny A bylo správně zařazeno 605 jedinců
# do skupiny B bylo správně zařazeno 54 jedinců
# do skupiny C bylo správně zařazeno 552 jedinců
# do skupiny D bylo správně zařazeno 593 jedinců

# Urči metriku accuracy pro rozhodovací strom a pro jeden ze dvou vybraných algoritmů
#print(accuracy_score(y_test, y_pred))
# accuracy_score = 0.4489795918367347, tedy 44,9%

scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf1 = KNeighborsClassifier()
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
#print(accuracy_score(y_test, y_pred1))
# accuracy_score = 0.41040318566450973, tedy 41,04% ==> lepší metriku měl rozhodovací strom

# Vyber cvik, který dle tebe nejvíce vypovídá o fyzické výkonnosti jedince. Porovnej, o kolik se zvýšila hodnota metriky accuracy pro oba algoritmy.
y = data["class"]
categorical_columns = ["gender"]
numeric_columns = ["age", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce", "sit-ups counts"]
numeric_data = data[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# rozhodovací strom, nové accuracy score
clf2 = DecisionTreeClassifier(max_depth=5)
clf2 = clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

#print(accuracy_score(y_test, y_pred2))
# accuracy_score = 0.5027376804380289, tedy 50,27% (původně 44,9%), metrika accuracy se zvýšila o 5,37%

#K Nearest Neighbours, nové accuracy score
y = data["class"]

scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = numpy.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf3 = KNeighborsClassifier()
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)
#print(accuracy_score(y_test, y_pred3))
# accuracy_score = 0.5164260826281732, tedy 51,64% (původně 41,04%), metrika accuracy se zvýšila o 10,6%