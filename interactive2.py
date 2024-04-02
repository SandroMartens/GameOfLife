import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget


class RGBArrayRenderer(QWidget):
    def __init__(self, rgb_array):
        super().__init__()
        self.original_rgb_array = rgb_array
        self.rgb_array = rgb_array.copy()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Slider for contrast adjustment
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(-100)
        self.slider.setMaximum(100)
        self.slider.setValue(0)  # Default value
        self.slider.valueChanged.connect(self.adjust_contrast)
        layout.addWidget(self.slider)

        # QLabel to display the QPixmap
        self.lbl = QLabel(self)
        layout.addWidget(self.lbl)

        # Convert the RGB array to a QImage
        self.update_image()

        self.setLayout(layout)
        self.setWindowTitle("RGB Array Renderer")
        self.setGeometry(300, 300, self.rgb_array.shape[1], self.rgb_array.shape[0])

    def update_image(self):
        height, width, channels = self.rgb_array.shape
        bytesPerLine = channels * width
        qImg = QImage(
            self.rgb_array.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qImg)
        self.lbl.setPixmap(pixmap)

    def adjust_contrast(self):
        # Adjust the contrast based on the slider's value
        # This is a simple linear contrast adjustment; more sophisticated methods exist
        factor = (259 * (self.slider.value() + 255)) / (
            255 * (259 - self.slider.value())
        )
        self.rgb_array = np.clip(
            128 + factor * (self.original_rgb_array - 128), 0, 255
        ).astype(np.uint8)
        self.update_image()


def main():
    app = QApplication(sys.argv)

    # Create a dummy RGB array for demonstration
    # Replace this with your actual RGB array
    rgb_array = np.random.randint(255, size=(480, 640, 3), dtype=np.uint8)

    ex = RGBArrayRenderer(rgb_array)
    ex.show()
    sys.exit(app.exec())


# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)
    rgb_array = np.random.randint(255, size=(480, 640, 3), dtype=np.uint8)
    ex = RGBArrayRenderer(rgb_array)
    ex.show()
    sys.exit(app.exec())
