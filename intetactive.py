import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget

from SmoothGOL import SmoothGameOfLife


class AnimatedLabel(QLabel):
    def __init__(self, width: int, height: int):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.width = width
        self.height = height
        self.frame_count = 0
        self.field_height = 200
        self.field_width = 200
        self.gol = SmoothGameOfLife(
            field_height=self.field_height,
            field_width=self.field_width,
            random_state=32,
        )
        self.gol.generate_initial_state()

        # Setup the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(100)  # Update interval in milliseconds
        self.resize(width, height)

    def update_animation(self):
        # Update your numpy array here
        self.update_numpy_array()

        # Convert numpy array to QImage and then to QPixmap
        image = QImage(
            self.data,
            self.field_width,
            self.field_height,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image).scaled(self.width, self.height)

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
        self.slider.setMinimum(0)  # Minimum interval of 1 ms
        self.slider.setMaximum(100)  # Maximum interval of 1000 ms
        self.slider.setValue(17)  # Default value
        self.slider.valueChanged.connect(self.update_interval)
        self.layout.addWidget(self.slider)

        # Create a QLabel to display the slider's current value
        self.sliderValueLabel = QLabel(f"k: {self.slider.value()}")
        self.layout.addWidget(self.sliderValueLabel)

    def update_interval(self, value):
        self.animatedLabel.gol.k = value / 100
        self.sliderValueLabel.setText(f"k: {value / 100}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = AnimationWidget(1000, 1000)
    mainWidget.show()
    sys.exit(app.exec())
