import cv2
import numpy as np

# Define a scale factor for display windows
scale_factor = 0.5

# Display function for scaling windows
def show_scaled_window(title, image, scale):
    resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    cv2.imshow(title, resized_image)

# Alignment function using SIFT and FLANN matching
def align_images(img1, img2):
    sift = cv2.SIFT_create()
    kp_img, desc_img = sift.detectAndCompute(img1, None)
    kp_img2, desc_img2 = sift.detectAndCompute(img2, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_img, desc_img2, k=2)
    good_p = []

    for m, n in matches:
        if m.distance < 0.85* n.distance:
            good_p.append(m)
    if len(good_p) > 4:
        q_p = np.float32([kp_img[m.queryIdx].pt for m in good_p]).reshape(-1, 1, 2)
        train_p = np.float32([kp_img2[m.trainIdx].pt for m in good_p]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(q_p, train_p, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img1, matrix, (img2.shape[1], img2.shape[0]))
        
        # Draw matches for visual confirmation
        match_img = cv2.drawMatches(img1, kp_img, img2, kp_img2, good_p, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        show_scaled_window("Feature Matches", match_img, scale_factor)
        
        return aligned_img
    else:
        print("Alignment failed.")
        return None

# Trackbar callback function (do nothing on callback)
def nothing(x):
    pass

# Create trackbars for color ranges
def create_color_trackbars(window_name, initial_values):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Lower H", window_name, initial_values[0], 180, nothing)
    cv2.createTrackbar("Lower S", window_name, initial_values[1], 255, nothing)
    cv2.createTrackbar("Lower V", window_name, initial_values[2], 255, nothing)
    cv2.createTrackbar("Upper H", window_name, initial_values[3], 180, nothing)
    cv2.createTrackbar("Upper S", window_name, initial_values[4], 255, nothing)
    cv2.createTrackbar("Upper V", window_name, initial_values[5], 255, nothing)

# Read trackbar positions
def get_trackbar_values(window_name):
    lower_h = cv2.getTrackbarPos("Lower H", window_name)
    lower_s = cv2.getTrackbarPos("Lower S", window_name)
    lower_v = cv2.getTrackbarPos("Lower V", window_name)
    upper_h = cv2.getTrackbarPos("Upper H", window_name)
    upper_s = cv2.getTrackbarPos("Upper S", window_name)
    upper_v = cv2.getTrackbarPos("Upper V", window_name)
    return np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v])

# Load and align images
base_image = cv2.imread('OneYearImage.jpg')
current_image = cv2.imread('coral1.jpg')
base_image = cv2.resize(base_image, (500, 500))
current_image = cv2.resize(current_image, (500, 500))

aligned_image = align_images(current_image, base_image)
if aligned_image is None:
    print("Alignment failed.")
    cv2.destroyAllWindows()
    exit()

# Show original images and aligned result
show_scaled_window("Base Image", base_image, scale_factor)
show_scaled_window("Current Image", current_image, scale_factor)
show_scaled_window("Aligned Image", aligned_image, scale_factor)

# Initial HSV values for pink and white
initial_pink = [140, 75, 50, 180, 255, 255]
initial_white = [0, 0, 200, 180, 66, 255]

# Create trackbars for pink and white masks
create_color_trackbars("Pink Mask", initial_pink)
create_color_trackbars("White Mask", initial_white)

# Mask generation function with dilation and erosion to remove noise
def create_mask(image, lower_bound, upper_bound):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Close small gaps
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)           # Dilation for noise reduction
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)            # Erosion to refine edges
    return mask

# Main loop for dynamic mask adjustment
while True:
    # Get current trackbar positions for pink and white masks
    lower_pink, upper_pink = get_trackbar_values("Pink Mask")
    lower_white, upper_white = get_trackbar_values("White Mask")

    # Generate individual color masks
    pink_mask_base = create_mask(base_image, lower_pink, upper_pink)
    white_mask_base = create_mask(base_image, lower_white, upper_white)
    pink_mask_aligned = create_mask(aligned_image, lower_pink, upper_pink)
    white_mask_aligned = create_mask(aligned_image, lower_white, upper_white)

    # Calculate masks for growth, death, cure, and infection
    growth_mask = cv2.bitwise_and(cv2.bitwise_or(white_mask_base, pink_mask_base), cv2.bitwise_not(cv2.bitwise_or(white_mask_aligned, pink_mask_aligned)))
    death_mask = cv2.bitwise_and(cv2.bitwise_or(white_mask_aligned, pink_mask_aligned), cv2.bitwise_not(cv2.bitwise_or(white_mask_base, pink_mask_base)))
    cure_mask = cv2.bitwise_and(pink_mask_aligned, white_mask_base)
    infection_mask = cv2.bitwise_and(white_mask_aligned, pink_mask_base)

    # Display each mask
    show_scaled_window("Base Image - Pink Mask", pink_mask_base, scale_factor)
    show_scaled_window("Base Image - White Mask", white_mask_base, scale_factor)
    show_scaled_window("Aligned Image - Pink Mask", pink_mask_aligned, scale_factor)
    show_scaled_window("Aligned Image - White Mask", white_mask_aligned, scale_factor)
    show_scaled_window("Growth Mask", growth_mask, scale_factor)
    show_scaled_window("Death Mask", death_mask, scale_factor)
    show_scaled_window("Cure Mask", cure_mask, scale_factor)
    show_scaled_window("Infection Mask", infection_mask, scale_factor)

    # Draw bounding boxes for growth, death, cure, and infection
    output_image = aligned_image.copy()

    def draw_contours(mask, image, color, min_area=1000):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    draw_contours(growth_mask, output_image, (0, 255, 0))       # Growth in green
    draw_contours(death_mask, output_image, (0, 0, 255))        # Death in red
    draw_contours(cure_mask, output_image, (255, 255, 255))     # Cure in white
    draw_contours(infection_mask, output_image, (255, 0, 255))  # Infection in pink

    # Display final output with all bounding boxes
    show_scaled_window("Annotated Detection", output_image, scale_factor)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up windows
cv2.destroyAllWindows()
