import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QGridLayout,
    QWidget,
    QPushButton,
)

from SmoothGOL import SmoothGameOfLife


class AnimatedLabel(QLabel):
    def __init__(
        self,
        width: int,
        height: int,
        field_height: int = 220,
        field_width: int = 220,
        cell_size: int = 3,
        init_density: float = 0.55,
        random_state: int = 32,
        timer_interval: int = 100,
    ):
        """
        Initializes the AnimatedLabel object with the specified width and height.
        Sets up the layout, frame count, field height, field width, and the SmoothGameOfLife object.
        Starts the QTimer for updating the animation.

        Args:
            width (int): The width of the label.
            height (int): The height of the label.
            field_height (int): The height of the field.
            field_width (int): The width of the field.
            cell_size (int): The size of each cell.
            init_density (float): The initial density of live cells.
            random_state (int): The random state for generating initial state.
            timer_interval (int): The interval for updating the animation in milliseconds.
        """
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.width = width
        self.height = height
        self.frame_count = 0
        self.field_height = field_height
        self.field_width = field_width
        self.gol = SmoothGameOfLife(
            field_height=self.field_height,
            field_width=self.field_width,
            cell_size=cell_size,
            init_density=init_density,
            random_state=random_state,
            # mode=1,
        )
        self.gol.generate_initial_state()

        # Setup the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(timer_interval)  # Update interval in milliseconds
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
        self.layout = QGridLayout(self)

        self.animatedLabel = AnimatedLabel(width, height)
        self.layout.addWidget(self.animatedLabel, 0, 0, 1, 2)

        sliders_params = [
            {"param_name": "k", "min_val": 1, "max_val": 100, "default_val": 18},
            {"param_name": "b1", "min_val": 0, "max_val": 100, "default_val": 19},
            {"param_name": "b2", "min_val": 0, "max_val": 100, "default_val": 44},
            {"param_name": "d1", "min_val": 0, "max_val": 100, "default_val": 31},
            {"param_name": "d2", "min_val": 0, "max_val": 100, "default_val": 44},
            {
                "param_name": "alpha_m",
                "min_val": 1,
                "max_val": 100,
                "default_val": 15,
            },
            {"param_name": "dt", "min_val": 0, "max_val": 100, "default_val": 100},
        ]
        self.sliders = {}
        for i, params in enumerate(sliders_params):
            self.create_slider(
                params["param_name"],
                params["min_val"],
                params["max_val"],
                params["default_val"],
                self.update_parameter,
                i,
            )

        # Create and setup the mode selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Mode 1")
        self.mode_selector.addItem("Mode 2")
        self.mode_selector.addItem("Mode 3")
        self.mode_selector.currentIndexChanged.connect(self.change_mode)
        self.layout.addWidget(self.mode_selector)  # Adjust position as needed

        # Create the reset button
        self.resetButton = QPushButton("Reset Simulation")
        self.resetButton.clicked.connect(self.reset_simulation)
        # Adjust the position in the layout as needed, for example, adding it below the mode selector
        self.layout.addWidget(
            self.resetButton, len(sliders_params) + 2, 0, 1, 2
        )  # Adjust grid position as needed

    def reset_simulation(self):
        """
        Resets the simulation to its initial state.
        """
        self.animatedLabel.gol.generate_initial_state()  # Assuming this method resets the game state
        self.animatedLabel.update_animation()  # Update the animation to reflect the reset state

    def change_mode(self, index):
        """
        Change the simulation mode based on the selected index from the dropdown.
        """
        self.animatedLabel.gol.mode = index + 1

    def create_slider(
        self,
        param_name: str,
        min_val: int,
        max_val: int,
        default_val: int,
        callback,
        row,
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
    mainWidget = AnimationWidget(800, 800)
    mainWidget.show()
    sys.exit(app.exec())
