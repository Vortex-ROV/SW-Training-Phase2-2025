import cv2
import numpy as np
import math


path = 'task 6/image2.jpeg'
img = cv2.imread(path)

# List to store points
points = []
scale_factor = None  
REFERENCE_LENGTH_CM = 32  

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def draw_line(event, x, y, flags, param):
    global scale_factor, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  

        if len(points) == 2:
            if scale_factor is None:
                reference_distance_px = calculate_distance(points[0], points[1])
                scale_factor = REFERENCE_LENGTH_CM / reference_distance_px
                print(f"Scale factor set: {scale_factor:.4f} cm per pixel")
                cv2.putText(img, "Scale set", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                distance_px = calculate_distance(points[0], points[1])
                distance_cm = distance_px * scale_factor
                print(f"Distance in cm: {distance_cm:.2f}")

                cv2.line(img, points[0], points[1], (0, 255, 0), 2)
                mid_x, mid_y = (points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2
                cv2.putText(img, f"{distance_cm:.2f} cm", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Image", img)
            points.clear()

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", draw_line)

while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
