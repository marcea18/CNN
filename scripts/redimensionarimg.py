import os
import cv2
ruta_dataset=r'/Users/marcea/Documents/PROYECTO_PLANTAS/CNN/dataset'
nueva_dimensiones=(128,128)
conjuentos=['train','validation','test']
for conjunto in conjuentos:
    ruta_conjunto=os.path.join(ruta_dataset,conjunto)
    especies=os.listdir(ruta_conjunto)
    for especie in especies:
        ruta_especie=os.path.join(ruta_conjunto,especie)
        imagenes=os.listdir(ruta_especie)
        for imagen in imagenes:
            ruta_imagen=os.path.join(ruta_especie,imagen)
            try:
                img=cv2.imread(ruta_imagen)
                img_redimensionada=cv2.resize(img,nueva_dimensiones,interpolation=cv2.INTER_AREA)
                #ruta_imagen_redimensionada=os.path.join(ruta_especie,f'resized_{imagen}')
                cv2.imwrite(ruta_imagen,img_redimensionada)
                print(f'Redimensionado: {ruta_imagen}')
            except Exception as e:
                print(f'Error redimensionando {ruta_imagen}: {str(e)}')
            


