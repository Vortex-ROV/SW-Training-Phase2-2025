import cv2
import numpy as np

# Load image
img = cv2.imread('image2.jpeg')

# Global array to store clicked points
points = []
ppc = None  # Pixels per centimeter (to be calculated)

def click_action(event, x, y, flags, param):
    global points, ppc
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # Display point on the image
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red dot to mark the clicked point
        cv2.imshow('Image', img)
        
        # Step 1: Set the reference object measurement
        if len(points) == 2 and ppc is None:
            # Calculate pixel distance of the reference object
            reference_pixel_distance = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
            
            # Known reference length in cm
            reference_length_cm = 32  # The reference object is 32 cm

            # Calculate pixels per cm (PPC)
            ppc = reference_pixel_distance / reference_length_cm
            print(f"Reference object: {reference_pixel_distance:.2f} pixels, PPC (Pixels per cm): {ppc:.2f}")

            # Draw green line for the reference object
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)  # Green line
            cv2.imshow('Image', img)

            # Clear points for next distance measurement
            points = []

        # Step 2: Measure distances after setting reference object
        elif len(points) == 2 and ppc is not None:
            # Calculate pixel distance between the two selected points
            pixel_distance = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
            
            # Convert pixel distance to cm using calculated PPC
            distance_cm = pixel_distance / ppc
            print(f"Distance in cm: {distance_cm:.2f}")
            
            # Draw blue line after calculating distance
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)  # Blue line for measurement
            cv2.imshow('Image', img)
            
            # Clear points for next measurement
            points = []

# Show image
cv2.imshow('Image', img)

# Mouse callback function
cv2.setMouseCallback('Image', click_action)

while True:
    # Wait for a key event for 1ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()
