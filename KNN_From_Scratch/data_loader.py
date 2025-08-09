from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd


def load_data(type_of_scale=None,test_size=0.2,val_size=0.2,random_state=42):
    #Incarcare dataset
    iris=load_iris()

    #Extrage caracteristici(features) si etichete(target)
    X=iris.data
    y=iris.target

    feature_names=iris.feature_names
    target_names=iris.target_names

    #Impartire date seturi de antrenament,validare si test
    X_train_val,X_test,y_train_val,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    val_size_relative=val_size/(1-test_size)
    X_train,X_val,y_train,y_val=train_test_split(X_train_val,y_train_val,test_size=val_size_relative,random_state=random_state)

    return X_train,X_val,X_test,y_train,y_val,y_test,feature_names,target_names



