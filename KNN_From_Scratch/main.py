from data_loader import load_data
from pipeline import create_knn_pipeline
from sklearn.metrics import classification_report

def main():
    #Load data
    X_train,X_val,X_test,y_train,y_val,y_test,feature_names,target_names=load_data()

    #Create and fit GridSearchCV
    knn_pipeline=create_knn_pipeline()
    knn_pipeline.fit(X_train,y_train)

    #Print best parameters
    print("Best parameter found:")
    print(knn_pipeline.best_params_)

    #Evaluate on validation test
    val_score=knn_pipeline.score(X_val,y_val)
    print(f"\nValidation accuracy:{val_score:.4f}")

    #Evaluate on test set
    test_score=knn_pipeline.score(X_test,y_test)
    print(f"Test accuracy:{test_score:.4f}")

    #Detailed classification report
    y_pred=knn_pipeline.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test,y_pred,target_names=target_names))

if __name__=='__main__':
    main()