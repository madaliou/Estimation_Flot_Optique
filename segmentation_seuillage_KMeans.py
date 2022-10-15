#SEUILLAGE AVEC KMEANS
import cv2 as cv
import numpy as np
import sys

seuil_1 = int(sys.argv[1])
seuil_2 = int(sys.argv[2])
K = int(sys.argv[3])

cap = cv.VideoCapture("dataset/Trafic2.avi")

ret, prevFrame = cap.read()

prev_gray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)

#Initialisation d'une matrice remplis de zeros ayant la même taille que le frame
mask = np.zeros_like(prevFrame)

# Nous mettons la saturation au maximum
mask[..., 1] = 255
lab = np.zeros_like(prevFrame)

while(cap.isOpened()):
  
  ret, currentFrame = cap.read()

  gray = cv.cvtColor(currentFrame, cv.COLOR_BGR2GRAY)

  shape = (currentFrame.shape[1], currentFrame.shape[0])
	
	# Calcul du flow optical dense avec la methode de Farneback
	#flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 15, 3, 5, 1.2, 0)
  flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,0.5, 3, 10, 5, 7, 1.5, 0)
	
	# Computes the magnitude and angle of the 2D vectors
  magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

  mask= cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
 
  attempts = 20
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  vectorized = np.float32(magnitude.reshape(-1, 1))
  ret, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
  pointCentral1 = np.uint8(center)
  pointCentral2 = np.uint8(center)

  lab[..., 0] = 0
  lab[..., 1] = 128
  lab[..., 2] = 128

  # Grande Vitesse d'objets 
  maximum = pointCentral1.max()
  for i, v in enumerate(pointCentral1):
      if v == [maximum]:
          pointCentral1[i] = 255
          pointCentral2[i] = 0
      else:
          pointCentral1[i] = 0

  # Vitesse moyenne d'objets
  maximum = pointCentral2.max()
  for i, v in enumerate(pointCentral2):
      if v == [maximum]:
          pointCentral2[i] = 255
      else:
          if pointCentral1[i] == 255:
              pointCentral2[i] = 255
          else:
              pointCentral2[i] = 0

  res_1 = pointCentral1[label.flatten()]
  lab[..., 1] = res_1.reshape((mask.shape))
  lab[..., 0] = res_1.reshape((mask.shape))
  lab[..., 1] = np.array([128 if value == 0 else value for value in lab[..., 1].reshape(shape[0] * shape[1])]).reshape(shape[1],shape[0])

  res_2 = pointCentral2[label.flatten()]
  lab[..., 2] = res_2.reshape((mask.shape))
  lab[..., 0] = res_2.reshape((mask.shape))
  lab[..., 2] = np.array([128 if value == 0 else value for value in lab[..., 2].reshape(shape[0] * shape[1])]).reshape(shape[1],
                                                                                                    shape[0])

  bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
  
  result = np.concatenate((currentFrame, bgr), axis=1)
  cv.imshow('original + bgr',result)
	
	# Mettre à jour le frame précedent avec le frame actuel
  prev_gray = gray
	
	# user presses the 'q' key
  if cv.waitKey(1) & 0xFF == ord('q'):	
    break

# closes all windows
cap.release()
cv.destroyAllWindows()