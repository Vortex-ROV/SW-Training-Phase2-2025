import cv2 as cv
import numpy as np
import math

start_point = None
end_point = None
drawing = False  
conversion_factor = None 
reference_length_cm = 32.0  

def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing, img_copy, conversion_factor

    if event == cv.EVENT_LBUTTONDOWN:
        if not drawing:  
            start_point = (x, y)
            drawing = True
        else:  
            end_point = (x, y)
            cv.line(img_copy, start_point, end_point, (0, 255, 0), 2)
            
            distance_px = calculate_distance(start_point, end_point)
            
            if conversion_factor is None:
                calibrate_image(distance_px)  # Set conversion factor based on the first drawn line
                print("Reference line set for calibration.")
            else:
                distance_cm = distance_px * conversion_factor
                midpoint = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
                cv.putText(img_copy, f'{distance_cm:.2f} cm', midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            start_point, end_point = None, None
            drawing = False

def calculate_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def calibrate_image(reference_pixel_length):
    global conversion_factor
    conversion_factor = reference_length_cm / reference_pixel_length
    print(f"Calibration complete: 1 pixel = {conversion_factor:.4f} cm")

img = cv.imread('calc_distance/image2.jpeg')
if img is None:
    print("Failed to load image")
    exit()

img_copy = img.copy() 

cv.namedWindow("Image")
cv.setMouseCallback("Image", draw_line)

while True:
    cv.imshow("Image", img_copy)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
