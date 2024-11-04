import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QStatusBar, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Detection System")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
        """)

        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.layout.addWidget(self.video_label)

        # Controls layout
        controls_layout = QHBoxLayout()

        # Start/Stop button
        self.toggle_button = QPushButton("Stop")
        self.toggle_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.toggle_button)

        self.layout.addLayout(controls_layout)

        # Status bar for face count
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

        self.detection_active = True

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.detection_active:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                # Detect faces with fixed parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Update the status bar with face count
                self.status_bar.showMessage(f"Detected Faces: {len(faces)}")

            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display the image
            self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio))

    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.toggle_button.setText("Stop" if self.detection_active else "Start")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
