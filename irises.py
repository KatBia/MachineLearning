from sklearn.externals import joblib
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


print(metrics.accuracy_score(y_test, y_pred))


sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
predictions = knn.predict(sample)
predict_species = [iris.target_names[p] for p in predictions]
print('predictions:', predict_species)


joblib.dump(knn, 'mlbrain.joblib')


model = joblib.load('mlbrain.joblib')
model.predict(X_test)
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
predictions = knn.predict(sample)
predict_species = [iris.target_names[p] for p in predictions]
print('predictions:', predict_species)
