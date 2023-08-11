import cv2
import os

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id=input('\n Enter user id and press Enter')
print("\n Initializing face capture ")
count=0
while(True):
    ret , img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)

    for(a,b,c,d) in faces :
        cv2.rectangle(img,(a,b),(a+c,b+d),(255,0,0),2)
        count+=1

        cv2.imwrite("dataset/User." +str(face_id)+"."+str(count)+".jpg",gray[b:b+d,a:a+c])
        cv2.imshow('image',img)

    k=cv2.waitKey(100) &0xff
    if k==27:
        
        break
    elif count>=30:
        break
print("\n Exiting program")
cam.release()
cv2.destroyAllWindows()
