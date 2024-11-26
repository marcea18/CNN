import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import matplotlib.pyplot as plt

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('../models/model_plantas_medicinales.h5')
    return model

# Cargar historial de entrenamiento
@st.cache_data
def load_history():
    with open('../models/history_model_plantas_medicinales.pkl', 'rb') as f:
        history = pickle.load(f)
    return history

# Interfaz de usuario
st.title("Clasificación de Plantas Medicinales 🌱")
st.sidebar.title("Opciones")

# Cargar el modelo y el historial
model = load_model()
history = load_history()

# Mostrar métricas del entrenamiento
if st.sidebar.checkbox("Mostrar métricas de entrenamiento"):
    st.subheader("Gráficas de entrenamiento")
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Precisión
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(epochs, history['val_accuracy'], label='Precisión Validación')
    plt.title("Precisión del modelo")
    plt.legend()
    st.pyplot(plt)

    # Pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['loss'], label='Pérdida Entrenamiento')
    plt.plot(epochs, history['val_loss'], label='Pérdida Validación')
    plt.title("Pérdida del modelo")
    plt.legend()
    st.pyplot(plt)

# Subir imágenes para predicción
st.subheader("Sube una imagen para clasificar")
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Procesar imagen
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar predicción
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    st.write(f"**Clase Predicha:** {class_index}")
    st.write(f"**Confianza:** {confidence:.2f}")
