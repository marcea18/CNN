import os
from PIL import Image
ruta_dataset=r'/Users/marcea/Documents/PROYECTO_PLANTAS/dataset'
formatos_validos=('.jpg','.jpeg','png','.bmp','.tiff')
def verificar_imagen(ruta_imagen):
    try:
        with Image.open(ruta_imagen) as img:
            img.verify()
        return True
    except Exception as e:
        print(f'Error al verificar la imagen: {e}')
        return False
#Recorrer las carpetas y verificar las imagenes
conjuntos=['train','validation','test']
for conjunto in conjuntos:
    ruta_conjunto=os.path.join(ruta_dataset,conjunto)
    especies=[d for d in os.listdir(ruta_conjunto) if os.path.isdir(os.path.join(ruta_conjunto,d))]
    for especie in especies:
        ruta_especie=os.path.join(ruta_conjunto,especie)
        imagenes=os.listdir(ruta_especie)
        for img_nombre in imagenes:
            ruta_imagen=os.path.join(ruta_especie,img_nombre)
            if not img_nombre.lower().endswith(formatos_validos):
                print(f"Archivo no válido (No es una imagen): {ruta_imagen}")
                os.remove(ruta_imagen)
            else:
                if not verificar_imagen(ruta_imagen):
                    print(f"Eliminar imagen inválida): {ruta_imagen}")    
                    os.remove(ruta_imagen)


