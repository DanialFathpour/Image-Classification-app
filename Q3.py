import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel , QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QSize
import tensorflow as tf
import tensorflow_hub as hub
import requests


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #Set window parameters----------
        self.HEIGHT = 600 
        self.WIDTH = 800
        self.setWindowTitle("image classifier")
        self.setGeometry(100, 100, self.WIDTH, self.HEIGHT)

        # Main window background color and font---------
        self.setStyleSheet("background-color: #f0f0f0; font-family: Arial;")

        # Video label-----------
        self.video_label = QLabel(self)
        self.video_label.setGeometry(20, 20, self.WIDTH*2//3, self.HEIGHT*2//3)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: white;")

        # Webcam capture and timer-------------------------
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

        # Buttons--------------
        self.capture_button = QPushButton(self)
        self.capture_button.setGeometry(self.WIDTH*3//4, self.HEIGHT*1//6, 150, 150)
        self.capture_button.setIcon(QIcon('Image-Capture-icon.png')) 
        self.capture_button.setIconSize(self.capture_button.size())
        self.capture_button.setStyleSheet("background-color: #f0f0f0; color: white; border: none; padding: 5px;")
        self.capture_button.clicked.connect(self.capture_frame)

        # Button to browse and select image file
        self.browse_button = QPushButton('Browse Image', self)
        self.browse_button.setGeometry(80, 480, 150, 60)
        self.browse_button.setStyleSheet("background-color: #007bff; color: white; border: none; padding: 5px;")
        self.browse_button.clicked.connect(self.browse_image)

        # Show and play buttons---------------
        self.show_frame_button = QPushButton("Show Frame", self)
        self.show_frame_button.setGeometry(self.WIDTH*3//4, self.HEIGHT*1//6 + 200, 230, 150)
        self.show_frame_button.setIcon(QIcon('show.png')) 
        self.show_frame_button.setIconSize(QSize(150, 150))
        self.show_frame_button.setStyleSheet("background-color: #f0f0f0; color: #f0f0f0; border: none; padding: 0px;border-radius: 25px;")
        self.show_frame_button.clicked.connect(self.show_captured_frame)

        # Classify button
        self.classify_button = QPushButton("Classify Image", self)
        self.classify_button.setGeometry(280, 480, 150, 60)
        self.classify_button.setStyleSheet("background-color: #28a745; color: white; border: none; padding: 5px;")
        self.classify_button.clicked.connect(self.classify_image)

        # Label to display classification result
        self.result_label = QLabel(self)
        self.result_label.setGeometry(490, 480, 250, 60)
        self.result_label.setStyleSheet("background-color: #ffffff; border: 2px solid black; padding: 5px;")
        self.result_label.setAlignment(Qt.AlignCenter)


        # Load the model
        self.model = hub.load("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5")
        self.labels = self.load_labels("https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt")
    
    def load_labels(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            labels = response.text.splitlines()
            if len(labels) > 1 and labels[0] == 'background':
                labels = labels[1:]  # remove background label if present
            return labels
        except requests.RequestException as e:
            print(f"Failed to fetch labels: {e}")
            return []

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(self.WIDTH*2//3, self.HEIGHT*2//3, Qt.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(p))

    def capture_frame(self):
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite('captured_frame.jpg', frame)
            cv2.destroyAllWindows()

    def browse_image(self):
        # Open file dialog to select an image file
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            if file_names:
                file_path = file_names[0]
                image = cv2.imread(file_path)
                cv2.imwrite('captured_frame.jpg', image)
        return image

    #show the captured frame in a seprate window-----------------
    def show_captured_frame(self):
        captured_frame = cv2.imread('captured_frame.jpg')
        cv2.imshow('Captured Frame', captured_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Classify the captured image
    def classify_image(self):
        image_path = 'captured_frame.jpg'
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        image = tf.expand_dims(image, axis=0)
        predictions = self.model(image)
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        if predicted_class < len(self.labels):
            predicted_label = self.labels[predicted_class - 1]
        else:
            predicted_label = "Unknown"
        self.result_label.setText(f'Predicted Label: {predicted_label}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

        