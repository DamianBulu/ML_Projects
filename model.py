from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted,check_X_y,check_array
from sklearn.base import BaseEstimator,ClassifierMixin
import numpy as np

class KNN(BaseEstimator,ClassifierMixin):
    def __init__(self,k=3,distance_metric='euclidean'):
        self.k=k
        self.distance_metric=distance_metric
    def compute_distance(self,a,b):
        if self.distance_metric=='euclidean':
            return np.sqrt(np.sum((a-b)**2))
        elif self.distance_metric=='manhattan':
            return np.sum(np.abs(a-b))
        else:
            raise ValueError("Invalid distance metric. Choose 'euclidean' or 'manhattan'")

    def fit(self,X,y):
        X,y=check_X_y(X,y)
        self.X_train=X
        self.y_train=y
        self.classes_=np.unique(y)
        return self
    def predict_one(self,x):
        distances=[self.compute_distance(x,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self,X):
        check_is_fitted(self) #Verify fit() has been called
        X=check_array(X)
        predictions=[self.predict_one(x) for x in X]
        return np.array(predictions)
    def evaluate(self,X,y):
        y_pred=self.predict(X)
        return accuracy_score(y,y_pred)
    def score(self,X,y):
        return accuracy_score(y,self.predict(X))

