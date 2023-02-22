import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

vid = cv.VideoCapture(0)
data = []

while True:
    ret, frame = vid.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    haarCascade = cv.CascadeClassifier(r'C:\Users\HP\OneDrive\Desktop\Python Projects\OpenCV Face Detection\haar_cascades.xml')
    faceDetect = haarCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 3
    )

    face = len(faceDetect)

    for (p,q,r,s) in faceDetect:
        cv.rectangle(frame, (p,q), (p+r,q+s), (0,0,255), 2)
        
        cv.putText(frame,
                   'Face',
                    (p,q-10),
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0,255,0), 
                    2
                    )
        
        slicedFace = frame[q:q+s, p:p+r, :]
        slicedFace = cv.resize(slicedFace, (50,50))
        
        if len(data) < 200:
            data.append(slicedFace)
        
        cv.imshow('Detected Face', frame)
           
    if cv.waitKey(20) & 0xFF == ord('q') or len(data) >= 200:
        print('Number of faces detected:', str(face)) 
        break

vid.release()
cv.destroyAllWindows()

np.save('without_mask_final.npy', data)
np.save('with_mask_final.npy', data)
