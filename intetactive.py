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

        pixmap = QPixmap.fromImage(image).scaled(
            self.width, self.height, mode=Qt.SmoothTransformation
        )

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

        self.sliders = {}
        self.create_slider("k", 0, 100, 17, self.update_parameter)
        # Example for another parameter
        self.create_slider("b1", 0, 100, 18, self.update_parameter)
        self.create_slider("b2", 0, 100, 43, self.update_parameter)
        self.create_slider("d1", 0, 100, 31, self.update_parameter)
        self.create_slider("d2", 0, 100, 43, self.update_parameter)
        self.create_slider("alpha_m", 0, 15, 18, self.update_parameter)

    def create_slider(self, param_name, min_val, max_val, default_val, callback):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        self.layout.addWidget(slider)

        # Create a QLabel to display the slider's current value
        sliderValueLabel = QLabel(f"{param_name}: {slider.value()}")
        self.layout.addWidget(sliderValueLabel)

        # Store the slider and its label in the dictionary
        self.sliders[param_name] = (slider, sliderValueLabel)

    def update_parameter(self, value):
        # Find out which slider was changed
        sender = self.sender()
        for param_name, (slider, label) in self.sliders.items():
            if sender == slider:
                # Update the parameter in AnimatedLabel
                setattr(self.animatedLabel.gol, param_name, value / 100)
                # Update the slider's label
                label.setText(f"{param_name}: {value / 100}")
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = AnimationWidget(800, 800)
    mainWidget.show()
    sys.exit(app.exec())
