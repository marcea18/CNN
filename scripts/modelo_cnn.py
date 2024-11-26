from tensorflow.keras import models, layers 

def crear_modelo(input_shape=(128,128,3), num_classes=14):
    model=models.Sequential()
    # primera capa convolucional
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(layers.MaxPool2D((2,2)))
    # Segunda capa convolucional
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    #Tercera capa convolucional
    model.add(layers.Conv2D(128,(3,3),activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    #Aplanamiento
    model.add(layers.Flatten())
    #Capas completamente conectadas
    model.add(layers.Dense(127,activation='relu'))
    #capa de salida
    model.add(layers.Dense(num_classes,activation='softmax'))
    return model
    
    
