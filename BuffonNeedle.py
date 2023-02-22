# Buffon's Needle in Python - Aditya Rajan

import cv2 as cv
import random
import numpy as np
import math
import time

crossList = []
pts = []
dist = 90
nLines = 6

lines = []

for i in range(0, nLines):
    lines.append((dist)*i+90)

poppedN = 0

throws = 0
x_throws = 50

mappedList = np.around(np.linspace(3.5,2.5,600,dtype=float), decimals = 3)

winLength = dist*(nLines+1)

img = np.zeros((winLength, winLength, 3), np.uint8)
font = cv.FONT_HERSHEY_SIMPLEX
ticks = [2.5,2.6,2.7,2.8,2.9,3,3.1,3.141,3.2,3.3,3.4,3.5]
labels = ['2.50','2.60','2.70','2.80','2.90','3.00','3.10','3.141','3.20','3.30','3.40','3.50']

start = time.process_time()
while True:
    throws+=1
    theta = math.radians(random.uniform(0,360))
    x1 = random.uniform(min(lines)-90, max(lines)+90)
    y1 = random.uniform(min(lines), max(lines))

    x2 = (x1+math.cos(theta)*(dist/2))
    y2 = (y1+math.sin(theta)*(dist/2))

    rx1, ry1, rx2, ry2 = (int(round(x1,0)),int(round(y1,0)),int(round(x2,0)),int(round(y2,0)))

    found = list(filter(lambda x: (y1 <= x <= y2) or (y2 <= x <= y1), lines))
    crossList.append(len(found)>0)

    pi_approx = throws/(sum(crossList) + 0.00001)
    pi_approx_y = round(pi_approx, 3)

    if (pi_approx_y>=2.50) and (pi_approx_y<=3.50):
        y_val = np.where(mappedList >= pi_approx_y)[0][-1]
    elif pi_approx_y < 2.50:
        y_val = int(0)
    elif pi_approx_y > 3.50:
        y_val = int(600)
    
    colourR = random.randint(0,255)
    colourG = random.randint(0,255)
    colourB = random.randint(0,255)

    cv.putText(img, 'Approx. Pi: '+ str(pi_approx), (int(winLength*0.779), 20), font, 0.6, (0,255,0), 2)
    cv.putText(img, "n = "+str(throws), (10,20), font, 0.5, (0, 255, 0), 2)

    cv.line(img, (rx1,ry1), (rx2,ry2), (colourB, colourG, colourR), 2)
    for line in lines:
        cv.line(img, (0,line), (winLength,line), (255,255,255), 2)

    cv.imshow("Approximation of pi using Buffon's needle problem", img)

    cv.rectangle(img, (int(winLength*0.75),0), (int(winLength), 60), (0,0,0), -1)
    cv.rectangle(img, (0,0), (100,40), (0,0,0), -1)

    if (throws%50 == 0):
        x_throws+=1
        pts.append((x_throws, y_val))
        img2 = np.zeros((winLength,winLength,3), np.uint8)

        cv.line(img2,(50,600),(winLength,600),(255,255,255),2)
        cv.line(img2,(50,0),(50,600),(255,255,255),2)

        cv.line(img2,(50,np.where(mappedList >= 3.141)[0][-1]),(winLength,np.where(mappedList >= 3.141)[0][-1]),(255,255,255),1)
        
        for tick, label in zip(ticks, labels):
            y_tick = np.where(mappedList>=tick)[0][-1]
            cv.line(img2, (47, y_tick), (53, y_tick), (255,255,255), 1)
            cv.putText(img2, label, (0, y_tick+3), font, 0.5, (255,255,255), 1)
        
        if x_throws>600:
            pts.pop(0)
            poppedN+=1

        for pt in pts:
            cv.rectangle(img2, (int(winLength*0.75), 0), (int(winLength), 60), (0,0,0), -1)
            cv.rectangle(img2, (0, 0), (100, 40), (0, 0, 0), -1)
            cv.putText(img2, "n = "+str(throws), (10,20), font, 0.5, (0, 255, 0), 2) ### (B,G,R)
            cv.putText(img2, 'Approx. Pi: '+ str(pi_approx), (int(winLength*0.779),20), font, 0.6, (0, 255, 0), 2)
            cv.circle(img2, (pt[0]-(poppedN),pt[1]), 1, (0, 255, 0), -1)
            cv.imshow('Graph', img2)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
    if throws == 100000:
        cv.destroyAllWindows()
        break
    
pi_final = throws/sum(crossList)
    
print(f'Number of needles: {throws}')  
print(f'Number of crosses: {sum(crossList)}')        
print(f'Overall approximation of pi: {pi_final}')
print('Time in seconds:',time.process_time()-start)

