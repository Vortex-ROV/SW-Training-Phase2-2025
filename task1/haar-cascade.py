import cv2 as cv 

cap = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier("C:\\Users\\mouha\\OneDrive\\Desktop\\SW-Training-Phase2-2025\\task1\\haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
