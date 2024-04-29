import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math 

# Initialize the video
cap = cv2.VideoCapture('Videos/vid (4).mp4')

# Create the color finder object 
# First true for finding hsvvalues then false
myColorFinder = ColorFinder(False) # true or false for debug mode
#hsvVals = 'red'
hsvVals = {'hmin':8,"smin":96,"vmin":115,"hmax":14,"smax":255,"vmax":255}
# Do expriemtn where only ball should be visible and then paste trackbar values to hsvVals
# mask is black and white version 

# Variables
posListX = []
posListY = []
xList = [i for i in range(0,1300)] # 1330 is width of image
prediction = False
while True:
    # Grap Image
    success, img = cap.read()
    #Read image 
    #img = cv2.imread("Ball.png")s
    #print(img)
    img = img[0:900,:] # height and width 

    # Find the Color Ball
    imgColor, mask = myColorFinder.update(img,hsvVals)
    # only ball color and mask shown

    # Find location of the ball 
    imgContours, contours = cvzone.findContours(img,mask,minArea=500) # area required to detect ball
    if contours:
        # 1st contour is biggest contour as they are sorted centerpoint
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])
    
    if posListX:
        # Polynominal Regression ( y = Ax^2 + Bx + C)
        # Find the coeffiencts
        A,B,C = np.polyfit(posListX,posListY,2) # quadratic has x2 , cubix x3


        for i,(posX,posY) in enumerate(zip(posListX,posListY)):
            pos = (posX,posY)
            cv2.circle(imgContours,pos,10,(0,0,255),cv2.FILLED)
            if i!=0:
                cv2.line(imgContours,pos,(posListX[i-1],posListY[i-1]),(0,255,0),5)
        
        for x in xList:
            y = int(A*x**2 + B*x + C)
            cv2.circle(imgContours,(x,y),2,(255,0,255),cv2.FILLED)
    
        if len(posListX)<10:
            # Prediction
            # X values are from 330 to 430
            # Y value is 590
            # Input Y and get x value
            #print(A,B,C)
            c = C - 590
            a = A
            b = B
            x = (-b - math.sqrt(b**2 - (4*a*c)))/(2*a)
            prediction = 330 < x < 430
        
        if prediction:
                cvzone.putTextRect(imgContours,"Basket",(50,190),scale=5,thickness=5,colorR=(0,200,0),offset=20)
        else:
                cvzone.putTextRect(imgContours,"No Basket",(50,190),scale=5,thickness=5,colorR=(0,0,200),offset=20)



    # Display
    imgContours = cv2.resize(imgContours,(0,0),None,0.7,0.7)
    #cv2.imshow('Image',img)
    cv2.imshow('ImageColor',imgContours)
    cv2.waitKey(100)

    #Open ball.png in mspaint / explorer
    # Find x and y value of the rim so check if ball in between 
    # Find x when ball at same height as rim 