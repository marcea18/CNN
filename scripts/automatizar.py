import os
import shutil
import random
#configuración
porcentaje_train=0.7
porcentaje_validation=0.15
porcentaje_test=0.15
ruta_dataset=r'/Users/marcea/Documents/PROYECTO_PLANTAS/CNN/datasetoriginal' #datasetoriginal
ruta_destino=r'/Users/marcea/Documents/PROYECTO_PLANTAS/CNN/dataset' #dataset
especies=os.listdir(ruta_dataset)#lista todos los nombre de carpetas o archivos
for especie in especies:
    ruta_especie=os.path.join(ruta_dataset,especie)
    imagenes=os.listdir(ruta_especie)
    random.shuffle(imagenes)#Mezcla aleatoriamente las imagenes
    num_total=len(imagenes)
    num_train=int(num_total*porcentaje_train)
    num_validation=int(num_total*porcentaje_validation)
    rutas ={'train':imagenes[:num_train],
            'validation':imagenes[num_train:num_train+num_validation],
            'test':imagenes[num_train+num_validation:]}
    
    for conjunto, imgs in rutas.items():
        ruta_conjunto= os.path.join(ruta_destino,conjunto,especie)
        os.makedirs(ruta_conjunto, exist_ok=True)
        for img in imgs:
            ruta_origen=os.path.join(ruta_especie,img)
            ruta_final=os.path.join(ruta_conjunto,img)
            shutil.copyfile(ruta_origen,ruta_final)





