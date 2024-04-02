import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel


class AnimatedLabel(QLabel):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.frame_count = 0

        # Initialize your numpy array with the desired shape and dtype
        self.data = np.zeros((height, width, 3), dtype=np.uint8)

        # Setup the timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(10)  # Update interval in milliseconds

    def update_animation(self):
        # Update your numpy array here
        # Example: Create a moving gradient
        self.data[:] = (
            self.frame_count % 255,
            self.frame_count * 2 % 255,
            self.frame_count * 3 % 255,
        )
        self.frame_count += 1

        # Convert numpy array to QImage and then to QPixmap
        image = QImage(self.data.data, self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        # Display the QPixmap in the QLabel
        self.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create an instance of the AnimatedLabel
    animated_label = AnimatedLabel(200, 200)
    animated_label.show()

    sys.exit(app.exec_())
