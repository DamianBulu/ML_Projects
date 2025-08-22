

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

def build_model(conv_filters=32,dense_units=128,learning_rate=0.001,dropout_rate=0.5,**kwargs):
    """Construiește un model CNN cu parametri configurabili"""
    model=Sequential([
        Conv2D(conv_filters,(3,3),activation='relu',input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(conv_filters*2,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(dense_units,activation='relu'),
        Dropout(dropout_rate),
        Dense(10,activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
