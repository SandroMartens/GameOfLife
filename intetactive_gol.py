import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QVBoxLayout, QWidget

from SmoothGOL import SmoothGameOfLife


class AnimatedLabel(QLabel):
    def __init__(self, width: int, height: int):
        """
        Initializes the AnimatedLabel object with the specified width and height.
        Sets up the layout, frame count, field height, field width, and the SmoothGameOfLife object.
        Starts the QTimer for updating the animation.

        Args:
            width (int): The width of the label.
            height (int): The height of the label.
        """
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
        """
        Updates the animation by updating the numpy array, converting it to a QImage and then to a QPixmap,
        and displaying the QPixmap in the QLabel.
        """
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
        """
        Updates the numpy array by calculating the next step of the SmoothGameOfLife object,
        applying the colormap to the array, and converting it to a numpy int8 array.
        """
        self.gol.calculate_next_step()
        rgb_array = self.gol.apply_colormap(self.gol.array)
        self.data = rgb_array.astype(np.int8)


class AnimationWidget(QWidget):
    """
    A QWidget subclass that displays an animated label and a set of sliders.
    Allows the user to interactively adjust the parameters of the animation.
    """

    def __init__(self, width: int, height: int):
        """
        Initializes the widget with the specified width and height.
        Creates an instance of the AnimatedLabel class and a set of sliders.
        """
        super().__init__()
        self.layout = QVBoxLayout(self)

        self.animatedLabel = AnimatedLabel(width, height)
        self.layout.addWidget(self.animatedLabel)

        sliders_params = [
            {"param_name": "k", "min_val": 0, "max_val": 100, "default_val": 17},
            {"param_name": "b1", "min_val": 0, "max_val": 100, "default_val": 18},
            {"param_name": "b2", "min_val": 0, "max_val": 100, "default_val": 43},
            {"param_name": "d1", "min_val": 0, "max_val": 100, "default_val": 31},
            {"param_name": "d2", "min_val": 0, "max_val": 100, "default_val": 43},
            {
                "param_name": "alpha_m",
                "min_val": 1,
                "max_val": 1000,
                "default_val": 1000,
            },
            {"param_name": "dt", "min_val": 0, "max_val": 100, "default_val": 99},
        ]
        self.sliders = {}
        for params in sliders_params:
            self.create_slider(
                params["param_name"],
                params["min_val"],
                params["max_val"],
                params["default_val"],
                self.update_parameter,
            )

    def create_slider(
        self, param_name: str, min_val: int, max_val: int, default_val: int, callback
    ):
        """
        Creates a slider with the specified parameters and adds it to the layout.
        Also creates a QLabel to display the slider's current value.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)
        self.layout.addWidget(slider)

        # Create a QLabel to display the slider's current value
        sliderValueLabel = QLabel(f"{param_name}: {slider.value()}")
        self.layout.addWidget(sliderValueLabel)

        # Store the slider and its label in the dictionary
        self.sliders[param_name] = (slider, sliderValueLabel)

    def update_parameter(self, value: int):
        """
        Updates the parameter of the AnimatedLabel object based on the value of the slider.
        """
        # Find out which slider was changed
        sender = self.sender()
        for param_name, (slider, label) in self.sliders.items():
            if sender == slider:
                # Update the parameter in AnimatedLabel
                setattr(self.animatedLabel.gol, param_name, value / 100)
                self.update_slider_label(param_name, value)
                break

    def update_slider_label(self, param_name: str, value: int):
        """
        Updates the label of the slider with the specified parameter name and value.
        """
        _, label = self.sliders[param_name]
        label.setText(f"{param_name}: {value / 100}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = AnimationWidget(1000, 600)
    mainWidget.show()
    sys.exit(app.exec())
