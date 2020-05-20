import cv2
import os
import time
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
images=os.listdir('images')
for image in images:
	imggr=cv2.imread('images/'+image,cv2.IMREAD_GRAYSCALE)
	img=cv2.imread('images/'+image)
	faces=face_cascade.detectMultiScale(imggr,3,4)
	i=0
	for (x,y,w,h) in faces:
		i=i+1
		print(f'resim_{i}')
		faceimg=img[y:y+h,x:x+w]
		tm="faces/"+str(int(time.time()))+".png"
		cv2.imwrite(tm, faceimg)
