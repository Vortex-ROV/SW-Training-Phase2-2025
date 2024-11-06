import cv2 as cv
import numpy as np

def aligning(img1, img2):
    # Initialize SIFT detector
    sift = cv.SIFT_create()  # Use cv.SIFT_create() if available
    kp_img, desc_img = sift.detectAndCompute(img1, None)  
    kp_img2, desc_img2 = sift.detectAndCompute(img2, None)  
    
    # Feature Matching using FLANN
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_img, desc_img2, k=2)
    
    good_p = []
    for m, n in matches:
        if m.distance < 0.591 * n.distance:
            good_p.append(m)
    
    if len(good_p) > 4:
        q_p = np.float32([kp_img[m.queryIdx].pt for m in good_p]).reshape(-1, 1, 2)
        train_p = np.float32([kp_img2[m.trainIdx].pt for m in good_p]).reshape(-1, 1, 2)
        
        matrix, mask = cv.findHomography(q_p, train_p, cv.RANSAC, 5.0)
        img_w = cv.warpPerspective(img1, matrix, (img1.shape[1], img1.shape[0]))
        
        return img_w  # Only returning aligned image

    return img1  # Return the original if alignment fails

# Load images
img1 = cv.imread('task 2/coral1.jpg')
cv.resize(img1,(600,360))
img2 = cv.imread('task 2/OneYearImage.jpg')
cv.resize(img2,(600,360))
# Align img1 to img2
img1_aligned = aligning(img1, img2)

# Convert aligned and target images to grayscale
# gray1 = cv.cvtColor(img1_aligned, cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Calculate the absolute difference
diff = cv.absdiff(img1, img2)
blurred_diff = cv.GaussianBlur(diff, (5, 5), 0)

# Threshold the difference image
_, thresh = cv.threshold(blurred_diff, 30, 255, cv.THRESH_BINARY)

# Dilate and erode to remove noise
kernel = np.ones((5, 5), np.uint8)
dilated = cv.dilate(thresh, kernel, iterations=2)
cleaned = cv.erode(dilated, kernel, iterations=1)

# Find edges and contours
edges = cv.Canny(cleaned, 100, 200)
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around larger contours
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 500:  # Filter small areas
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img1_aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
result = np.hstack((img1_aligned, img2))
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()