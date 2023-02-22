# Facemask Detector using cv2 and Support Vector Machine (SVM) Algorithm, by Aditya Rajan
# To add: Probabilities for Mask and No Mask.
# To change: Overfitting issues, must shuffle testing data.

import cv2 as cv
import numpy as np
from sklearn.svm import SVC      # i.e Support Vector Classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA      # i.e Principal Component Analysis (used for Dimensionality Reduction)

withMask = np.load('with_mask_final.npy')
withoutMask = np.load('without_mask_final.npy')

withMask = withMask.reshape(200, 50*50*3)
withoutMask = withoutMask.reshape(200, 50*50*3)

print(withMask.shape)
print(withoutMask.shape)

X = np.r_[withMask, withoutMask]

labels = np.zeros(X.shape[0])
labels[200:] = 1.0

names ={0 : 'Mask', 1 : 'No Mask'}

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train) # in this method, PCA uses eigenvalues and eigenvectors (look into that).
x_test = pca.fit_transform(x_test)
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

acc = int(accuracy_score(y_test, y_pred))*100
print('Accuracy: ', acc, '%' )
if acc == 100:
    print('NOTE: Model may be overfitting!')


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

    for (p,q,r,s) in faceDetect:
        cv.rectangle(frame, (p,q), (p+r,q+s), (0,0,255), 2)
        
        
        slicedFace = frame[q:q+s, p:p+r, :]
        slicedFace = cv.resize(slicedFace, (50,50))
        slicedFace = slicedFace.reshape(1,-1)

        pred = svm.predict(slicedFace)
        heading = names[int(pred)]

        cv.putText(frame, heading ,(p,q-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow('Detected Face', frame)
           
    if cv.waitKey(20) & 0xFF == ord('q') or len(data):
        break

vid.release()
cv.destroyAllWindows()
