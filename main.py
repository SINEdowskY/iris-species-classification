from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = load_iris()

iris_data = iris_dataset['data']
iris_target = iris_dataset['target']

X_train, X_test, y_train, y_test = train_test_split(
    iris_data, iris_target, random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Wynik dla zestawu danych testowych: {:.2f}".format(knn.score(X_test, y_test)))



