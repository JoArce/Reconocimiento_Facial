from cv2 import cv2 as cv
import os

dataRuta = "C:/Cursos/Reconocimiento_Facial/Data"
listaData = os.listdir(dataRuta)

entrenamientoFaceEigenRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoFaceEigenRecognizer.read(
    "C:/Cursos/Reconocimiento_Facial/EntrenamientoEigenRecognizer.xml")
ruidos = cv.CascadeClassifier(
    "C:/Cursos/Reconocimiento_Facial/haarcascade_frontalface_default.xml")
camara = cv.VideoCapture(1)
while True:
    _, captura = camara.read()
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura = grises.copy()
    cara = ruidos.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in cara:
        rostroCapturado = idCaptura[y:y + e2, x:x + e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamientoFaceEigenRecognizer.predict(rostroCapturado)
        cv.putText(captura, '{}'.format(resultado), (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv.LINE_AA)
        if resultado[1] > 6500:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x, y - 20), 2, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
        else:
            cv.putText(captura, "No encontrado", (x, y - 20), 2, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)

    cv.imshow("Resultados", captura)
    if cv.waitKey(1) == ord('s'):
        break

camara.release()
cv.destroyAllWindows()