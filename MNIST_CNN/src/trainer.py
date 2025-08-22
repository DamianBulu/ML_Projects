
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from data_loader import load_and_preprocess_data
from model_builder import build_model
import joblib

def train_model():
    """Antrenează modelul final cu parametrii optimi"""
    X_train,X_val,y_train,y_val=load_and_preprocess_data()

    # Încarcă parametrii optimi sau folosește default
    try:
        best_params=joblib.load('outputs/models/best_params.pkl')
        print("Folosind parametrii optimi găsiți")
    except:
        best_params={
            'conv_filters':32,
            'dense_units' :128,
            'learning_rate': 0.001,
            'dropout_rate' :0.5
        }
        print("Folosind parametrii default")
    # Construiește modelul
    model=build_model(**best_params)

    #Callbacks
    callbacks=[
        EarlyStopping(patience=3,monitor='val_loss'),
        ModelCheckpoint('outputs/models/final_model.h5',save_best_only=True)
    ]

    #Antrenare
    history=model.fit(
        X_train,y_train,
        validation_data=(X_val,y_val),
        epochs=50,
        batch_size=best_params.get('batch_size',64),
        callbacks=callbacks,
        verbose=1
    )

    return history

if __name__=='__main__':
    train_model()

