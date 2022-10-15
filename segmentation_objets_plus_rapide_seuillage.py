import numpy as np
import cv2 as cv
import sys

seuil = int(sys.argv[1])

#Lecture de la vidéo 
cap = cv.VideoCapture("dataset/Venice.avi")

#Découpage de la vidéo en gardant le premier frame
ret, prevFrame = cap.read()
shape = (prevFrame.shape[1], prevFrame.shape[0])
#Nous convertissons le premier frame en niveau de gris
prevFrameGray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
#Initialisation d'une matrice remplis de zeros ayant la même taille que le frame
hsv = np.zeros_like(prevFrame)

hsv[..., 1] = 255
hsv[..., 0] = 255

while (1):
    ret, currentFrame = cap.read()
    if not ret:
        #s'il n'ya plus de frame
        break
    next = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)
    #Calcul du flot optique
    flow = cv.calcOpticalFlowFarneback(prevFrameGray, next, None, 0.5, 3, 10, 5, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    #Segmentation des objets les plus rapides basée sur le seuillage 
    hsv[..., 0] = 255
    ret, hsv[..., 0] = cv.threshold(hsv[..., 2], seuil, 255, cv.THRESH_BINARY)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    height, width, _ = np.array(currentFrame).shape
    #Affichage du résultat avec comparaison avec l'original
    result = np.concatenate((currentFrame, bgr), axis=1)

    cv.imshow('Resultat', result)

    # Appuyer sur 'q' pour quitter le programme
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prevFrameGray = next

cap.release()

cv.destroyAllWindows()