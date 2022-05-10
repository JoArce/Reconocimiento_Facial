from cv2 import cv2 as cv, waitKey
from cv2 import COLOR_BGR2GRAY
import os

# carpeta donde se guardas las fotos que saco de los videos o con la camara
# modelo = 'Fotos Usuario'
# modelo = 'Fotos Auron'
modelo = 'Fotos Elon'
ruta1 = "C:/Cursos/Reconocimiento_Facial"  # ruta donde se guardaran mis imagenes
rutaCompleta = ruta1 + '/' + modelo
if not os.path.exists(
        rutaCompleta):  # si hemos olvidado crear la ruta aqui nos crea la ruta con las variables que ya declaramos previamente
    os.makedirs(rutaCompleta)

# camara = cv.VideoCapture(1) # con esto usamos la camara del celular para sacar las fotos
# camara  = cv.VideoCapture("C:/Cursos/Reconocimiento_Facial/videoauron.mp4") #video de auron
camara = cv.VideoCapture("C:/Cursos/Reconocimiento_Facial/ElonMusk.mp4")
ruido = cv.CascadeClassifier(
    "C:/Cursos/Reconocimiento_Facial/haarcascade_frontalface_default.xml")
id = 0

while True:
    respuesta, captura = camara.read()
    if respuesta == False: break

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = captura.copy()

    caras = ruido.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in caras:
        cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
        rostroCapturado = idCaptura[y:y + e2, x:x + e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutaCompleta + '/imagen_{}.jpg'.format(id),
                   rostroCapturado)
        id = id + 1


    cv.imshow("Resultado rostro", captura)

    if id == 11:
        break

camara.release()
cv.destroyAllWindows()
