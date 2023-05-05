import cv2 as cv
import numpy as np

class ObjectDetector():
    """
        Clase para la deteccion de objetos

        Atributos
        -------
        kernel : ndarray
            Color RGB del objeto detectado como una matriz con la forma (1,3)
        threshold : int
            Umbral de pertenencia de un punto a un objeto

        Methods
        -------
        detect(img)
            Especifica el rectángulo delimitador del objeto con el color dado
    """

    def __init__(self):
        color = [0,127,255] # azul
        self.kernel = np.asarray([[color]])
        self.threshold = 200

    def detect(self, img):
        """
            Especifica el rectangulo delimitador del objeto con el color dado

            Deteccion basada en coincidencia de color 

            Parametros
            ----------
            img : ndarray
                Entrada de imagen de 3 canales con modelo de color RGB

            Retorno
            -------
            int
                Coordenada horizontal
            int
                Coordenada vertical
            int
                Ancho
            int
                Altura
        """

        map = 255 - np.mean(np.abs((img - self.kernel)), axis=2)
        _, map = cv.threshold(map.astype(np.uint8), 200, 255, cv.THRESH_BINARY)
        return cv.boundingRect(map)