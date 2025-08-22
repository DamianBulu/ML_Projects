import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from scikeras.wrappers import KerasClassifier
from data_loader import load_and_preprocess_data
from model_builder import build_model

def tune_hyperparameters():
    """Execută randomized search pentru găsirea parametrilor optimi"""
    X_train,_,y_train,_=load_and_preprocess_data()
    #Definim spatiul de cautare
    param_dist={
        'conv_filters':[32,64,128],
        'dense_units':[64,128,256],
        'learning_rate':loguniform(1e-4,1e-2),
        'dropout_rate':[0.3,0.5,0.7],
        'batch_size':[32,64,128]
    }

    #Cream modelul pt tuning
    #model=KerasClassifier(build_fn=build_model,epochs=3,verbose=1)
    model = KerasClassifier(
        model=build_model,
        epochs=2,
        verbose=1,
        conv_filters=32,
        dense_units=128,
        learning_rate=0.001,
        dropout_rate=0.5
    )

    #Randomized search
    search=RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=4, #numar de combinatii testate
        cv=2, #3-fold cross validation
        verbose=2,
        random_state=42
    )

    #Executa cautarea pe un subset de date(pt viteza)
    search_result=search.fit(X_train[:10000],y_train[:10000])

    #Salveaza cel mai bun model
    best_model = search_result.best_estimator_.model_
    best_model.save('outputs/models/best_model.h5')

    #Salveaza parametrii optimi
    joblib.dump(search_result.best_params_,'outputs/models/best_params.pkl')

    print(f'Cel mai bun scor:{search_result.best_score_:.4f}')
    print(f'Cei mai buni parametri:{search_result.best_params_}')

    return search_result.best_params_

if __name__=="__main__":
    tune_hyperparameters()
