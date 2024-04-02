import sys

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget
from SmoothGOL import SmoothGameOfLife


class AnimatedLabel(QLabel):
    def __init__(self, width, height):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.width = width
        self.height = height
        self.frame_count = 0
        self.gol = SmoothGameOfLife(
            screen_height=height, screen_width=width, square_size=1
        )
        self.gol.generate_initial_state()

        # Initialize your numpy array with the desired shape and dtype
        self.data = np.zeros((height, width, 3), dtype=np.uint8)

        # Setup the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(10)  # Update interval in milliseconds
        self.resize(width, height)

    def update_animation(self):
        # Update your numpy array here
        self.update_numpy_array()

        # Convert numpy array to QImage and then to QPixmap
        image = QImage(self.data, self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Display the QPixmap in the QLabel
        self.setPixmap(pixmap)

    def update_numpy_array(self):
        self.gol.calculate_next_step()
        rgb_array = self.gol.apply_colormap(self.gol.array)
        self.data = rgb_array.astype(np.int8)


class AnimationWidget(QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.animatedLabel = AnimatedLabel(width, height)
        self.layout.addWidget(self.animatedLabel)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)  # Minimum interval of 1 ms
        self.slider.setMaximum(1000)  # Maximum interval of 1000 ms
        self.slider.setValue(10)  # Default value
        self.slider.valueChanged.connect(self.update_interval)
        self.layout.addWidget(self.slider)

    def update_interval(self, value):
        self.animatedLabel.timer.setInterval(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = AnimationWidget(200, 200)
    mainWidget.show()
    sys.exit(app.exec())
