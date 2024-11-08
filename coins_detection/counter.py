from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model with local weights
model = YOLO("best.pt")  

# Perform inference on a local image
results = model.predict(source="coins2.jpg", conf=0.4)  

# Initialize counters for each class
one_pound_count = 0
half_pound_count = 0
quarter_pound_count = 0

# Iterate over each prediction and count based on class
for obj in results[0].boxes.data:
    class_id = int(obj[5])  
    if class_id == 0: 
        half_pound_count += 1
    elif class_id == 1:  
        one_pound_count += 1
    elif class_id == 2:  
        quarter_pound_count += 1

# Calculate the total sum
total_sum = (one_pound_count * 1 + half_pound_count * 0.5 + quarter_pound_count * 0.25)

# Print the counts for each class
print("One Pound Count:", one_pound_count)
print("Half Pound Count:", half_pound_count)
print("Quarter Pound Count:", quarter_pound_count)
print("Total Sum:", total_sum)

# Save the annotated image with predictions
annotated_img = results[0].plot()
cv2.imwrite("prediction.jpg", annotated_img)

# Display the annotated image
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
