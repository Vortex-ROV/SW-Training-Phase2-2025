import numpy as np# imports numpy library
import cv2# imports opencv library that deals with imges
green=(0,255,0)# rgb value for green note their order is BGR not  RGB
red=(0,0,255)# rgb value for red
blue=(255,0,0)# rgb value for blue
yellow=(0,255,255)# rgb value for yellow

#lower_white = np.array([93, 14, 120])# set lower RGB values to detect the white color between them and the upper values in the frame
#upper_white = np.array([153, 84, 255])# set upper RGB values to detect the white color between them and the lower values in the frame
#lower_pink = np.array([0, 56, 0])# set lower RGB values to detect the pink color between them and the upper values in the frame
#upper_pink = np.array([179, 255, 255])# set upper RGB values to detect the pink color between them and the lower values in the frame

#lower_white_new = np.array([93, 14, 120])# set lower RGB values to detect the white color between them and the upper values in the frame
#upper_white_new = np.array([153, 84, 255])# set upper RGB values to detect the white color between them and the lower values in the frame
#lower_pink_new = np.array([0, 56, 0])# set lower RGB values to detect the pink color between them and the upper values in the frame
#upper_pink_new = np.array([179, 255, 255])# set upper RGB values to detect the pink color between them and the lower values in the frame
cv2.namedWindow("Trackbars")
cv2.namedWindow("Trackbars_new")
cv2.namedWindow("Trackbars_new_pink")
cv2.namedWindow("Trackbars_pink")
def nothing():# make a function that passes so it gets out of the case and do nothing whenever this funtion is called
    pass# pass and get out of the case

cv2.createTrackbar("L - H", "Trackbars", 93, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 14, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 120, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 153, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 84, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)



cv2.createTrackbar("L - H-n", "Trackbars_new", 93, 179, nothing)
cv2.createTrackbar("L - S-n", "Trackbars_new", 14, 255, nothing)
cv2.createTrackbar("L - V-n", "Trackbars_new", 120, 255, nothing)
cv2.createTrackbar("U - H-n", "Trackbars_new", 153, 179, nothing)
cv2.createTrackbar("U - S-n", "Trackbars_new", 84, 255, nothing)
cv2.createTrackbar("U - V-n", "Trackbars_new", 255, 255, nothing)


cv2.createTrackbar("L - H-p", "Trackbars_pink", 0, 179, nothing)
cv2.createTrackbar("L - S-p", "Trackbars_pink", 56, 255, nothing)
cv2.createTrackbar("L - V-p", "Trackbars_pink", 0, 255, nothing)
cv2.createTrackbar("U - H-p", "Trackbars_pink", 179, 179, nothing)
cv2.createTrackbar("U - S-p", "Trackbars_pink", 255, 255, nothing)
cv2.createTrackbar("U - V-p", "Trackbars_pink", 255, 255, nothing)

cv2.createTrackbar("L - H-p-n", "Trackbars_new_pink", 0, 179, nothing)
cv2.createTrackbar("L - S-p-n", "Trackbars_new_pink", 56, 255, nothing)
cv2.createTrackbar("L - V-p-n", "Trackbars_new_pink", 0, 255, nothing)
cv2.createTrackbar("U - H-p-n", "Trackbars_new_pink", 179, 179, nothing)
cv2.createTrackbar("U - S-p-n", "Trackbars_new_pink", 255, 255, nothing)
cv2.createTrackbar("U - V-p-n", "Trackbars_new_pink", 255, 255, nothing)





def aligning (img1,img2):# defines a functions that applies feature recognition to two imges
    sift = cv2.SIFT_create()# make sift feature recognition
    kp_img, desc_img = sift.detectAndCompute(img1, None)  # Key Points and descreptors of the first img note descreptors are
    #  the variable that holds the details of the features
    kp_img2, desc_img2 = sift.detectAndCompute(img2, None)  # Key Points and descreptors of the second img
    # Feature Matching
    index_params = dict(algorithm=0, trees=5)# the indexes that we will be searching in
    search_params = dict()# the search parameters of our search
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_img, desc_img2, k=2)# try to find the matches between the decriptors in he first img  and the second one
    good_p= []  # list (array)for good matches (good points)

    for m, n in matches:# makes a for loop so that m and n have value of matches respectivly so m is for the first img and is for the second
        if m.distance < 0.591 * n.distance:# if the distance between them is low (low error) [[[[[[0.591]]]]] works for both imgs the best
            good_p.append(m)#append this values  to an array
    if len(good_p) > 4:# if there is more than 4 matches
        q_p = np.float32([kp_img[m.queryIdx].pt for m in good_p]).reshape(-1, 1, 2)  # query points are the points of the first img
        train_p = np.float32([kp_img2[m.trainIdx].pt for m in good_p]).reshape(-1, 1, 2)# train points are the points of the second img
        matrix, mask = cv2.findHomography(q_p, train_p, cv2.RANSAC, 5.0)# find the homography of this imge or you can say we finf the amount
        # we are going to move the img to be perfectly aligned with the new img
        img_w = cv2.warpPerspective(img1,matrix,(img1.shape[1],img1.shape[0]))#move  old picture to match the new one[apply prespective transformation]
        matches_mask = mask.ravel().tolist()# get coordinates of the points this line is optional
        # Perspective transform
        img3 = cv2.drawMatches(img1, kp_img, img2, kp_img2, good_p, img2)# creat an img that draws the matches between both the old and new picture
    return img_w,img3# return the old img after being shifted to be perfectly aligned with the new one and the img that draw the matches

def masking(hsv,lower,upper):# defines a funtion that takes 3 variables to do masking on them
    # the first is hsv the second is lower range the third is upper range
    mask=cv2.inRange(hsv,lower,upper)# detects a mask from hsv if it is in the range of lower and upper
    return mask# returns mask as our output

def draw(msk,clr):# define a function that draws a rectangle around the msk with clr as the color of the rectangle
    contours,_ = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# detect the contours of the mask

    for contour in contours:# give contour all the values of contours
        area = cv2.contourArea(contour)# make the variable area holds the area value of the contours of the msk

        if area > 1000:# if the area of the detected contours is > 1000 pixel
            x, y, w, h = cv2.boundingRect(contour)# get a rectangle around these detected contours
            cv2.rectangle(new, (x, y), (x + w, y + h), clr, 2)# draw this recangle on those points with color clr

        else:# if it not >1000
            nothing()# do nothing (wont draw any thing)

new = cv2.imread("task 2/coral1.jpg")# read the old imge from the computer
old = cv2.imread("task 2/OneYearImage.jpg")#read the second img from the computer

l_h = cv2.getTrackbarPos("L - H", "Trackbars")
l_s = cv2.getTrackbarPos("L - S", "Trackbars")
l_v = cv2.getTrackbarPos("L - V", "Trackbars")
u_h = cv2.getTrackbarPos("U - H", "Trackbars")
u_s = cv2.getTrackbarPos("U - S", "Trackbars")
u_v = cv2.getTrackbarPos("U - V", "Trackbars")



l_h_n = cv2.getTrackbarPos("L - H-n", "Trackbars_new")
l_s_n = cv2.getTrackbarPos("L - S-n", "Trackbars_new")
l_v_n = cv2.getTrackbarPos("L - V-n", "Trackbars_new")
u_h_n = cv2.getTrackbarPos("U - H-n", "Trackbars_new")
u_s_n = cv2.getTrackbarPos("U - S-n", "Trackbars_new")
u_v_n = cv2.getTrackbarPos("U - V-n", "Trackbars_new")




l_h_p = cv2.getTrackbarPos("L - H-p", "Trackbars_pink")
l_s_p = cv2.getTrackbarPos("L - S-p", "Trackbars_pink")
l_v_p = cv2.getTrackbarPos("L - V-p", "Trackbars_pink")
u_h_p = cv2.getTrackbarPos("U - H-p", "Trackbars_pink")
u_s_p = cv2.getTrackbarPos("U - S-p", "Trackbars_pink")
u_v_p = cv2.getTrackbarPos("U - V-p", "Trackbars_pink")

l_h_n_p = cv2.getTrackbarPos("L - H-p-n", "Trackbars_new_pink")
l_s_n_p = cv2.getTrackbarPos("L - S-p-n", "Trackbars_new_pink")
l_v_n_p = cv2.getTrackbarPos("L - V-p-n", "Trackbars_new_pink")
u_h_n_p = cv2.getTrackbarPos("U - H-p-n", "Trackbars_new_pink")
u_s_n_p = cv2.getTrackbarPos("U - S-p-n", "Trackbars_new_pink")
u_v_n_p = cv2.getTrackbarPos("U - V-p-n", "Trackbars_new_pink")



lower_white = np.array([l_h, l_s, l_v])# set lower RGB values to detect the white color between them and the upper values in the frame
upper_white = np.array([u_h, u_s, u_v])# set upper RGB values to detect the white color between them and the lower values in the frame

lower_white_new = np.array([l_h_n, l_s_n, l_v_n])# set lower RGB values to detect the white color between them and the upper values in the frame
upper_white_new = np.array([u_h_n, u_s_n, u_v_n])# set upper RGB values to detect the white color between them and the lower values in the frame


lower_pink = np.array([l_h_p, l_s_p, l_v_p])# set lower RGB values to detect the pink color between them and the upper values in the frame
upper_pink = np.array([u_h_p, u_s_p, u_v_p])# set upper RGB values to detect the pink color between them and the lower values in the frame

lower_pink_new = np.array([l_h_n_p, l_s_n_p, l_v_n_p])# set lower RGB values to detect the pink color between them and the upper values in the frame
upper_pink_new = np.array([u_h_n_p, u_s_n_p, u_v_n_p])#

frs = old.shape#get its shape in a tupelq
frs1 = new.shape#get its shape in a tupel
# new = cv2.resize(new ,(400,600))# give the new img  width 400 and hight 600
# old = cv2.resize(old ,(400,600))# give the old img  width 400 and hight 600
img33,matchings_img = aligning(old,new)# say that the input variabels of the aligning func which are img1 and img2 are old and  new and the return value
# will equal  to img33 and the second return value will equal matching imge
hsv_old_homo=cv2.cvtColor(img33, cv2.COLOR_BGR2HSV)# transform all the rgb values of the old picture after being aligned into HSV values
hsv_new=cv2.cvtColor(new,cv2.COLOR_BGR2HSV)# transform all the rgb values of the new picture into HSV values

mask_white_old_homo=masking(hsv_old_homo,lower_white,upper_white)# says that the return value of the function masking is mask_white_old_homo and that our hsv vlaue
# which is the first in the function is hsv old  and ower lower and upper ranges are lower white and upper white

mask_pink_old_homo=masking(hsv_old_homo,lower_pink,upper_pink)# the same but pink instead white

mask_white_new=masking(hsv_new,lower_white_new,upper_white_new)# says that the return value of the function masking is mask_white_old_homo and that our hsv vlaue
# which is the first in the function is hsv new  and ower lower and upper ranges are lower white and upper white

mask_pink_new=masking(hsv_new,lower_pink_new,upper_pink_new)# the same but pink instead of white


po_dif_pn=cv2.subtract(mask_pink_old_homo,mask_pink_new)# you subtract mask_pink_new from mask_pink_old_homo----mask_pink_old_homo-mak_pink_new
# the result will be a mask that have the output of either death or bleech or both
pn_dif_po=cv2.subtract(mask_pink_new,mask_pink_old_homo)# the difrenc between mask pink new and mask pink old homo  will get you either growth or recovery or both
wo_dif_wn=cv2.subtract(mask_white_old_homo,mask_white_new)# the difrence between mask white old homo and mask white new will get us death or recovery
wn_dif_wo=cv2.subtract(mask_white_new,mask_white_old_homo)# .......................will get us growth or bleech

pn_dif_wo=cv2.subtract(mask_pink_new,mask_white_old_homo)# this will get us the difrence between pink new and white old which is the new pink without the recovery
po_dif_wn=cv2.subtract(mask_pink_old_homo,mask_white_new)# if we extracted from the old pink the newwhite then its like removing the bleeching because it doesnt see the rest of the new white
# but those who ere once pink so they  are present in the old pink

growth_p=cv2.subtract(pn_dif_wo,mask_pink_old_homo)# ....... if we extrracted from the new pink without the recovery the old pink we will get our growth as an output
#but this growth is the growth of a neew pink branch not all growth
death_p=cv2.subtract(po_dif_wn,mask_pink_new)# if we subtracted from the old pink the new white[the branches that were pink and turned white]the new pink
# then we will get our pink death if it is present
bleech=cv2.subtract(po_dif_pn,death_p)#as we stated that po_dif_pn will  get death_p or bleech or both of them so if we extracted the death_p from it
# we will get our bleech if it is present
recovery=cv2.subtract(pn_dif_po,growth_p)#as we stated that pn_dif_po will  get growth_p or recovery or both of them so if we extracted the growth_p from it
# we will get our recovery if it is present
growth_w=cv2.subtract(wn_dif_wo,bleech)#as we stated that wn_dif_wo will  get growth_w or bleech or both of them so if we extracted the bleech from it
# we will get our growth_w if it is present
death_w=cv2.subtract(wo_dif_wn,recovery)#as we stated that wo_dif_wn will  get death_w or recovery or both of them so if we extracted the recovery from it
# we will get our death_w if it is present
growth=growth_w+growth_p# says that the growth is both growth in white and growth in pink so we add them[growth_w+growth_p] together
death=death_w+death_p# says that the death is both death in white and death in pink so we add them[death_w+death_p] together

draw(growth,green)# call the funtion hat draws rectangle over its input mask  with its defined function and we define our input mask as growth and the clr of the rectangle
#green
draw(death,yellow)# call the funtion hat draws rectangle over its input mask  with its defined function and we define our input mask as death and the clr of the rectangle
#yellow
draw(recovery,blue)# call the funtion hat draws rectangle over its input mask  with its defined function and we define our input mask as recovery mask and the clr of the rectangle
#blue
draw(bleech,red)# call the funtion hat draws rectangle over its input mask  with its defined function and we define our input mask as bleech and the clr of the rectangle
#red

#cv2.imshow("growth_w",growth_w)# shows the white growth mask with string "growth_w"
cv2.imshow("img_new", new)# shows the new imgwith string "img_new"
cv2.imshow("immg_old", img33)# shows the old img after being aligned with the new one  with string "imggg3"
#cv2.imshow("recovery",recovery)# shows the recoverymask with string "recovery"
#cv2.imshow("growth_p",growth_p)# shows the growth_p mask with string "growth_p"
#cv2.imshow("death_p",death_p)# shows the death_p mask with string "death_p"
#cv2.imshow("bleech",bleech)# shows the bleech mask with string "bleech"
#cv2.imshow("death_w",death_w)# shows the death_w mask with string "death_w"
#cv2.imshow("total_growth",growth)# shows the total_growth mask with string "total_growth"
#cv2.imshow("total_death",death)# shows the total_death mask with string "total_death"
#cv2.imshow("img3", matchings_img)  # show the img that draws the  matches

key = cv2.waitKey(1)
cv2.waitKey(0)
cv2.destroyAllWindows()