import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QFrame
)


class ShapeDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape Detector")
        self.setGeometry(100, 100, 800, 600)

        # Define color ranges in HSV
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]),  # Red needs two ranges
            'green': ([40, 100, 100], [80, 255, 255], None, None),
            'blue': ([100, 100, 100], [140, 255, 255], None, None),
            'yellow': ([20, 100, 100], [40, 255, 255], None, None)
        }

        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left panel for camera feed
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Camera feed label with a border
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.camera_label.setLineWidth(2)
        left_layout.addWidget(self.camera_label)

        # Right panel for detection status
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Status labels for each shape
        self.status_labels = {}
        shapes = ["Rectangle", "Triangle", "Circle", "Star"]
        for shape in shapes:
            label = QLabel(f"{shape}: Not Detected")
            label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    font-size: 14px;
                }
            """)
            self.status_labels[shape] = label
            right_layout.addWidget(label)

        # Control buttons
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)

        right_layout.addWidget(self.start_button)
        right_layout.addWidget(self.stop_button)
        right_layout.addStretch()

        # Add panels to the main layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

        # Initialize detection components
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def handle_shape_detection(self, shape_name, color="#ffcccc"):
        """Updates the status label for detected shapes"""
        self.status_labels[shape_name].setText(f"{shape_name}: DETECTED!")
        self.status_labels[shape_name].setStyleSheet(
            f"background-color: {color}; padding: 10px; border-radius: 5px;"
        )

    def reset_status_labels(self):
        """Resets all status labels to their default state"""
        for shape, label in self.status_labels.items():
            label.setText(f"{shape}: Not Detected")
            label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")

    def get_color_mask(self, hsv_frame, color):
        """Create a mask for the specified color"""
        lower1, upper1, lower2, upper2 = self.color_ranges[color]
        mask1 = cv2.inRange(hsv_frame, np.array(lower1), np.array(upper1))

        if lower2 is not None:  # For red, which needs two ranges
            mask2 = cv2.inRange(hsv_frame, np.array(lower2), np.array(upper2))
            return cv2.bitwise_or(mask1, mask2)
        return mask1

    def detect_shapes(self, frame):
        """Detect shapes in the frame based on color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Combine masks for all colors
        for color in self.color_ranges.keys():
            color_mask = self.get_color_mask(hsv, color)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue

            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Reduced epsilon for more precise approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                continue

            # Check if the shape is in the center region of the frame
            height, width = frame.shape[:2]
            center_region = (
                    width // 4 < cX < 3 * width // 4 and
                    height // 4 < cY < 3 * height // 4
            )

            if not center_region:
                continue

            # Draw the contour
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

            # Identify shapes based on vertices and properties
            vertices = len(approx)
            shape_name = "Unknown"

            if vertices == 3:
                shape_name = "Triangle"
                self.handle_shape_detection("Triangle", "#ccffcc")

            elif vertices == 4:
                shape_name = "Rectangle"
                self.handle_shape_detection("Rectangle", "#ffcccc")

            elif 8 <= vertices <= 9:  # Relaxed circle detection
                # Check circularity
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if circularity > 0.8:
                    shape_name = "Circle"
                    self.handle_shape_detection("Circle", "#ffffcc")

            elif 9 <= vertices <= 12:
                # Calculate the center point of all vertices
                center_x = np.mean([p[0][0] for p in approx])
                center_y = np.mean([p[0][1] for p in approx])

                # Calculate distances from center to all points
                distances = [np.sqrt((p[0][0] - center_x) ** 2 + (p[0][1] - center_y) ** 2) for p in approx]

                # Calculate the ratio between max and min distances
                max_dist = max(distances)
                min_dist = min(distances)
                ratio = min_dist / max_dist if max_dist > 0 else 0

                # If there is significant variation in distances (characteristic of a star)
                if 0.3 <= ratio <= 0.7:  # Adjusted ratio range for star points
                    shape_name = "Star"
                    self.handle_shape_detection("Star", "#ccccff")

            # Add text label
            cv2.putText(frame, shape_name, (cX - 20, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def update_frame(self):
        """Process each frame for shape detection"""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        # Reset status labels at the start of each frame
        self.reset_status_labels()

        # Detect shapes
        processed_frame = self.detect_shapes(frame)

        # Convert and display the frame
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio))

    def start_detection(self):
        """Start the camera and detection process"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.timer.start(30)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_detection(self):
        """Stop the camera and detection process"""
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_label.clear()
            self.reset_status_labels()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = ShapeDetectionGUI()
    window.show()
    sys.exit(app.exec_())
