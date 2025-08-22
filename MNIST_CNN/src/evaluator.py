
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_and_preprocess_data

def evaluate_model():
    """Evaluare și vizualizare performanță model"""
    _,X_test,_,y_test=load_and_preprocess_data()

    #incarca cel mai bun model
    try:
        model=load_model('outputs/models/best_model.h5')
        print('Evaluare model optim gasit prin hyperparameter tuning')
    except:
        model=load_model('outputs/models/final_model.h5')
        print('Evaluare model final antrenat')

    #Evaluare pe setul de test
    test_loss,test_acc=model.evaluate(X_test,y_test,verbose=0)
    print(f"Test accuracy:{test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    #Vizualizare arhitectura
    model.summary()

if __name__=="__main__":
    evaluate_model()
