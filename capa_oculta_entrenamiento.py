from cv2 import cv2 as cv
import os
import numpy as np
from time import time

dataRuta = "C:/Cursos/Reconocimiento_Facial/Data"
listaData = os.listdir(dataRuta)
ids = []
rostrosData = []
id = 0

tiempoInicial = time()
for  fila in listaData:
    rutaCompleta = dataRuta +'/'+ fila
    for archivo in os.listdir(rutaCompleta):
        print('Imagen :', fila +'/' + archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta + '/' + archivo,0))

    id = id +1
    tiempoFinalLectura = time()
    tiempoTotalLectura = tiempoFinalLectura - tiempoInicial
    print(f'Tiempo Lectura: {tiempoTotalLectura}')

entrenamientoEigenRecognizer = cv.face.EigenFaceRecognizer_create()
print('Iniciando el entrenamiento........ por favor espere, gracias.')
entrenamientoEigenRecognizer.train(rostrosData, np.array(ids))

entrenamientoEigenRecognizer.write("EntrenamientoEigenRecognizer.xml")
print('Entrenamiento concluido')