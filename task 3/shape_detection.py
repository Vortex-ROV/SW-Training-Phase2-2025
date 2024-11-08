import cv2
import numpy  as np
img =cv2.imread('task 3/clean_6269.jpg')
img=cv2.resize(img,(500,500))
def nothing(x):
    pass
cv2.namedWindow('track bars')
cv2.createTrackbar('lower_h','track bars',0,179,nothing)
cv2.createTrackbar('lower_s','track bars',47,255,nothing)
cv2.createTrackbar('lower_v','track bars',75,255,nothing)

cv2.createTrackbar('upper_h','track bars',61,179,nothing)
cv2.createTrackbar('upper_s','track bars',255,255,nothing)
cv2.createTrackbar('upper_v','track bars',255,255,nothing)

img=cv2.GaussianBlur(img,(5,5),0)

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
while True:
    l_h=cv2.getTrackbarPos('lower_h','track bars')
    l_s=cv2.getTrackbarPos('lower_s','track bars')
    l_v=cv2.getTrackbarPos('lower_v','track bars')
    u_h=cv2.getTrackbarPos('upper_h','track bars')
    u_s=cv2.getTrackbarPos('upper_s','track bars')
    u_v=cv2.getTrackbarPos('upper_v','track bars')
    
    lower=np.array([l_h,l_s,l_v])
    upper=np.array([u_h,u_s,u_v])
    
    mask=cv2.inRange(img_hsv,lower,upper)
    kernel=np.ones((5,5),dtype='uint8')

    mask = cv2.dilate(mask,kernel,iterations=5)
    mask  =cv2.Canny(mask,75,150)
    cnts=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2]
    for cnt in cnts:
        area=cv2.contourArea(cnt)
        if area >1500:
            hull=cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)

            x=approx.ravel()[0]
            y=approx.ravel()[1]
            if len(approx)==3:
                cv2.putText(img,'triangle',(x,y),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0))
            if len(approx)==4:
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                (xs, ys), (w, h), ang = cv2.minAreaRect(approx)
                if w+25>h and h>w-25:
                    cv2.putText(img,'square',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0))
                else:
                    cv2.putText(img,'rectangle',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0))
            if len(approx)==5:
                cv2.putText(img,'star',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0))
            if len(approx)>6 and len(approx)<13:
                cv2.putText(img,'circle',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0))
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            
            
    cv2.imshow('mask',mask)
    cv2.imshow('img',img)
    cv2.imwrite('shapes.jpg',img)
    if cv2.waitKey(1)&0xFF ==ord('s'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()