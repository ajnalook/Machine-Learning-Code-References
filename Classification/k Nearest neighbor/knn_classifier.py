from sklearn.neighbors import KNeighborsClassifier


class KNN_CLASSIFIER:
    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors=3)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        predictions = self.clf.predict(X)
        return predictions



if __name__ == '__main__':
    ## Dummy Data Generation
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    # Initialize Classifier
    clf = KNN_CLASSIFIER()
    clf.fit(X,Y)
    # Get predictions
    print(clf.predict(X))
