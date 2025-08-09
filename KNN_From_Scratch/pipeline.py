from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from model import KNN


def create_knn_pipeline():
    pipeline=Pipeline([
        ('scaler',None),
        ('knn',KNN())
    ])

    param_grid=[
        {
            'scaler':[None,StandardScaler(),MinMaxScaler()],
            'knn__k':range(1,20,2),
            'knn__distance_metric':['euclidean','manhattan']
        }
    ]

    #Create GridSearchCV object

    grid_search=GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    return  grid_search

