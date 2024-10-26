import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

while True:
    ret,frame = cap.read()
    if ret:
        frame_grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_grey)
        eyes = eye_cascade.detectMultiScale(frame_grey)

        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


        for x,y,w,h in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(5) == ord('q'):
            break

cv2.destroyAllWindows()