from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from modelo_cnn import crear_modelo
import matplotlib.pyplot as plt
#Configuración
ruta_entrenamiento=r'/Users/marcea/Documents/PROYECTO_PLANTAS/CNN/dataset/train'
ruta_validation=r'/Users/marcea/Documents/PROYECTO_PLANTAS/CNN/dataset/validation'
batch_size=32
epochs=14
num_classes=14
#crear modelo

model = crear_modelo( num_classes = num_classes)
#compilar el modelo
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Generador de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
) 
validation_datagen=ImageDataGenerator(rescale=1./255)

#Generadores
train_generator=train_datagen.flow_from_directory(
    ruta_entrenamiento,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator=validation_datagen.flow_from_directory(
    ruta_validation,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode='categorical'    
)

#Entrenar el modelo
history=model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples// batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//batch_size,
    epochs=epochs)


#guardar el modelo
model.save('../models/model_plantas_medicinales.h5')

#guardar el historial de entrenamiento
import pickle
with open('../models/history_model_plantas_medicinales.pkl','wb') as f:
    pickle.dump(history.history, f)
    