#SEUILLAGE AVEC DIFFERENTE VITESSES
import cv2 as cv
import numpy as np
import sys
#Utilisation des paramètres prises en entrée saisis par l'utilisateur
seuil_1 = int(sys.argv[1])
seuil_2 = int(sys.argv[2])


cap = cv.VideoCapture("dataset/Trafic2.avi")

ret, prevFrame = cap.read()

prev_gray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)

mask = np.zeros_like(prevFrame)

# Nous mettons la saturation de l'image au maximum
mask[..., 1] = 255
lab = np.zeros_like(prevFrame)

while(cap.isOpened()):
	
	ret, currentFrame = cap.read()

	gray = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)

	shape = (currentFrame.shape[1], currentFrame.shape[0])
	
	#flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 15, 3, 5, 1.2, 0)
	flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 10, 5, 7, 1.5, 0)
	
	# Computes the magnitude and angle of the 2D vectors
	magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
 
	mask= cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

	lab[..., 0] = 0

	lab[..., 1] = 128
	
	lab[..., 2] = 128
	# Segmentation des objets de différentes vitesses basée sur le seuillage
	ret, lab[..., 0] = cv.threshold(mask, seuil_2, 255, cv.THRESH_BINARY)

	ret, lab[..., 1] = cv.threshold(mask, seuil_2, 255, cv.THRESH_BINARY)
	lab[..., 1] = np.array([128 if value == 0 else value for value in lab[..., 1].reshape(shape[0] * shape[1])]).reshape(shape[1], shape[0])
	ret, lab[..., 0] = cv.threshold(mask, seuil_1, 255, cv.THRESH_BINARY)
	ret, lab[..., 2] = cv.threshold(mask, seuil_1, 255, cv.THRESH_BINARY)
	lab[..., 2] = np.array([128 if value == 0 else value for value in lab[..., 2].reshape(shape[0] * shape[1])]).reshape(shape[1],	shape[0])
	
	#Conversion de LAB à BGR
	bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
	
	result = np.concatenate((currentFrame, bgr), axis=1)
	#Affichage du résultat
	cv.imshow('original + bgr',result)

	#Mettre à jour le frame précédent
	prev_gray = gray

	if cv.waitKey(1) & 0xFF == ord('q'):
		
		break

# closes all windows
cap.release()
cv.destroyAllWindows()
