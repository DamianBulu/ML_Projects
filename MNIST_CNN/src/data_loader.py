
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    (X_train,y_train),(X_test,y_test)=mnist.load_data()

    #Normalizare si redimensionare
    X_train=X_train.astype('float32')/255.0
    X_test=X_test.astype('float32')/255.0
    X_train=np.expand_dims(X_train,axis=-1)
    X_test=np.expand_dims(X_test,axis=-1)

    #One-hot encoding
    y_train=to_categorical(y_train,10)
    y_test=to_categorical(y_test,10)

    return X_train,X_test,y_train,y_test
