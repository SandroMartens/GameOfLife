import sys
from functools import partial
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QSlider,
    QGridLayout,
    QWidget,
    QPushButton,
)


class SmoothGameOfLife:
    """SmoothLife — continuous generalisation of Conway's Game of Life (Rafler 2011).

    The grid stores a real-valued state function f(x, t) in [0, 1].  Each step
    computes two local averages for every cell x:
      - inner filling  m  (eq. 1): mean of f over a disk of radius ri = cell_size
      - outer filling  n  (eq. 2): mean of f over the annulus ri < r < ra, ra = 3*ri

    The transition function s(n, m) (eq. 6) uses smooth sigmoid intervals to
    decide birth and survival, mixing them by the sigmoid of m (aliveness).
    Time is advanced via eq. 8:  f += dt * delta(s(n, m)).
    """

    def __init__(
        self,
        colormap: str = "magma",
        dt: float = 1,
        random_state: int | None = None,
        survival_lower_threshold: float = 0.1875,
        survival_upper_threshold: float = 0.4375,
        birth_lower_threshold: float = 0.3125,
        birth_upper_threshold: float = 0.4375,
        field_width: int = 200,
        field_height: int = 200,
        alpha_m: float = 0.15,
        cell_size: float = 1,
        init_density: float = 0.5,
        mode: int = 1,
        k: float = 0.18,
    ) -> None:
        """
        Parameters
        ----------
        dt : float
            Time step size for eq. 8.  dt=1 gives discrete stepping; smaller
            values yield smoother continuous time evolution (Section 4).
        survival_lower_threshold, survival_upper_threshold : float
            Boundaries [d1, d2] of the survival (death) interval in eq. 6.
        birth_lower_threshold, birth_upper_threshold : float
            Boundaries [b1, b2] of the birth interval in eq. 6.
        alpha_m : float
            Steepness alpha_m of the sigmoid sigma_1 applied to the inner
            filling m (eq. 5).  Controls how sharply aliveness transitions
            around m = 0.5.
        cell_size : float
            Inner radius ri of the disk (eq. 1).  The outer radius is ra = 3*ri.
        k : float
            Steepness alpha_n of the sigmoid sigma_1 applied to the outer
            filling n (eqs. 3–4).
        init_density : float
            Fraction of cells initialised with a non-zero value.
        mode : int
            Selects the delta formula for time-stepping (eq. 8):
            1 – s(n,m) − f  (difference from current state)
            2 – 2*s(n,m) − 1  (paper's smooth time-stepping)
            3 – s(n,m) − m  (difference from inner filling)
        """
        self.field_width = field_width
        self.field_height = field_height
        self.colormap = colormap
        self.init_density = init_density
        self.dt = dt
        self.birth_lower_threshold = birth_lower_threshold
        self.birth_upper_threshold = birth_upper_threshold
        self.survival_lower_threshold = survival_lower_threshold
        self.survival_upper_threshold = survival_upper_threshold
        self.cell_size = cell_size
        self.alpha_m = alpha_m
        self.mode = mode
        self.random_state = random_state
        self.k = k

    def generate_initial_state(self) -> None:
        """Randomise f(x, 0) in [0, 1] with the given density of live cells."""
        rng = np.random.default_rng(self.random_state)
        array = rng.random(size=(self.field_width, self.field_height))
        mask = rng.random(size=array.shape) < self.init_density
        self.current_state_array = np.where(mask, array, 0)

    def calculate_next_step(self) -> None:
        """Advance the state by one time step dt using eqs. 6 and 8.

        1. Compute inner filling m (eq. 1) and outer filling n (eq. 2) via
           Gaussian convolutions with ri and ra = 3*ri.
        2. Subtract the inner contribution from the neighbourhood sum to
           approximate the true annular average.
        3. Compute aliveness sigma_1(m, 0.5) with steepness alpha_m (eq. 5).
        4. Evaluate birth and survival sigmoid intervals sigma_2 (eq. 4).
        5. Mix birth/survival by aliveness to obtain s(n, m).
        6. Update f via f += dt * delta  (eq. 8).
        """
        cell_radius = self.cell_size
        neighborhood_radius = 3 * cell_radius

        neighbor_sums = self._convolve(
            radius=neighborhood_radius, size=5 * neighborhood_radius
        ).clip(0, 1)
        cell_sums = self._convolve(radius=cell_radius, size=5 * cell_radius).clip(0, 1)

        # Subtract inner disk contribution to approximate the annular average n.
        neighbor_sums = (neighbor_sums - 1 / 9 * cell_sums) / (8 / 9)

        # Aliveness: sigma_1(m, 0.5) with steepness alpha_m (eq. 5).
        aliveness = self._sigmoid(k=self.alpha_m, x=cell_sums, offset=0.5)

        birth = self._interval(
            neighbor_sums, self.birth_lower_threshold, self.birth_upper_threshold
        )
        survival = self._interval(
            neighbor_sums, self.survival_lower_threshold, self.survival_upper_threshold
        )

        # Transition function s(n, m): mix birth/survival by aliveness (eq. 6).
        next_step = (aliveness * survival + (1 - aliveness) * birth).clip(0, 1)
        dx = self._delta(cell_sums, next_step)
        self.current_state_array = (self.current_state_array + self.dt * dx).clip(0, 1)

    def _delta(self, cell_sums: np.ndarray, next_step: np.ndarray) -> np.ndarray:
        """Return the rate of change dx for the time update f += dt * dx.

        Three formulations from Rafler (2011) Section 4 and Millington (2012):

        Mode 1 — exponential relaxation (Millington):
            dx = s(n,m) − f
            Pushes f towards s exponentially; tends to be stable.

        Mode 2 — paper formulation (Rafler eq. 8):
            dx = 2·s(n,m) − 1
            Maps s ∈ [0,1] to [−1,1] as a signed rate of change.
            Can diverge when cells are in a stable state.

        Mode 3 — inner-filling relaxation (Millington):
            dx = s(n,m) − m
            Variant of Mode 1 with the inner filling m substituted for f.
        """
        if self.mode == 1:
            return next_step - self.current_state_array
        elif self.mode == 2:
            return 2 * next_step - 1
        else:
            return next_step - cell_sums

    def _interval(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Smooth interval indicator sigma_2(x, lo, hi) (eq. 4).

        Returns sigma_1(x, lo) * (1 − sigma_1(x, hi)), i.e. a value near 1
        when lo < x < hi and near 0 outside, with smooth transitions of
        steepness k (alpha_n in the paper).
        """
        return self._sigmoid(self.k, x, lo) - self._sigmoid(self.k, x, hi)

    def _convolve(self, radius: float, size: float) -> np.ndarray:
        """Approximate the circular average integral (eqs. 1–2) via a Gaussian kernel.

        The paper uses a disk/annulus kernel with anti-aliasing (Section 3).
        Here a Gaussian with sigma=radius serves as a differentiable substitute.
        Periodic boundary conditions (mode='wrap') model a toroidal grid.
        """
        kernel = self._gaussian_kernel(sigma=radius, size=size)
        return scipy.ndimage.convolve(self.current_state_array, kernel, mode="wrap")

    def _gaussian_kernel(self, size: float, sigma: float) -> np.ndarray:
        """Normalised 2-D Gaussian kernel used to approximate disk integrals."""
        half = int(size) // 2
        x, y = np.mgrid[-half : half + 1, -half : half + 1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()

    def _sigmoid(self, k: float, x: np.ndarray, offset: float) -> np.ndarray:
        """Sigmoid sigma_1(x, a) with steepness parameter k (eq. 3).

        sigma_1(x, a) = 1 / (1 + exp(-(x - a) * 4 / k))
        """
        return 1 / (1 + np.exp(-4 / k * (x - offset)))

    def apply_colormap(self, array: np.ndarray) -> np.ndarray:
        """Map state values in [0, 1] to RGB using the chosen matplotlib colormap."""
        cmap = plt.get_cmap(self.colormap)
        return cmap(array)[:, :, :3] * 255


class SliderConfig(NamedTuple):
    name: str
    min_val: int
    max_val: int
    default_val: int


SLIDER_CONFIGS = [
    SliderConfig("k", 1, 100, 18),
    SliderConfig("birth_lower_threshold", 0, 100, 31),
    SliderConfig("birth_upper_threshold", 0, 100, 44),
    SliderConfig("survival_lower_threshold", 0, 100, 19),
    SliderConfig("survival_upper_threshold", 0, 100, 44),
    SliderConfig("alpha_m", 1, 100, 15),
    SliderConfig("dt", 0, 100, 100),
]


class AnimatedLabel(QLabel):
    """QLabel that runs the SmoothLife simulation and renders each frame."""

    def __init__(
        self,
        window_width: int,
        window_height: int,
        field_height: int = 300,
        field_width: int = 300,
        cell_size: int = 1,
        init_density: float = 0.7,
        random_state: int = 32,
        timer_interval: int = 100,
    ):
        super().__init__()
        self.field_height = field_height
        self.field_width = field_width
        self.gol = SmoothGameOfLife(
            field_height=field_height,
            field_width=field_width,
            cell_size=cell_size,
            init_density=init_density,
            random_state=random_state,
        )
        self.gol.generate_initial_state()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(timer_interval)
        self.resize(window_width, window_height)
        self._display_width = window_width
        self._display_height = window_height

    def update_animation(self):
        """Advance the simulation by one step and redraw the label."""
        self.gol.calculate_next_step()
        rgb_array = self.gol.apply_colormap(self.gol.current_state_array)

        image = QImage(
            rgb_array.astype(np.uint8).tobytes(),
            self.field_width,
            self.field_height,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image).scaled(
            self._display_width,
            self._display_height,
            mode=Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class AnimationWidget(QWidget):
    """Main window: simulation display plus interactive parameter sliders."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self._layout = QGridLayout(self)

        self.animatedLabel = AnimatedLabel(width, height)
        self._layout.addWidget(self.animatedLabel, 0, 0, 1, 2)

        self.slider_widgets: dict[str, tuple[QSlider, QLabel]] = {}
        for row, config in enumerate(SLIDER_CONFIGS):
            self._add_slider(config, row + 1)

        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Mode 1", "Mode 2", "Mode 3"])
        self.mode_selector.currentIndexChanged.connect(
            lambda i: setattr(self.animatedLabel.gol, "mode", i + 1)
        )
        self._layout.addWidget(self.mode_selector, len(SLIDER_CONFIGS) + 1, 0, 1, 2)

        reset_button = QPushButton("Reset Simulation")
        reset_button.clicked.connect(self._reset_simulation)
        self._layout.addWidget(reset_button, len(SLIDER_CONFIGS) + 2, 0, 1, 2)

    def _add_slider(self, config: SliderConfig, row: int) -> None:
        """Create a labelled slider for one simulation parameter."""
        label = QLabel(f"{config.name}: {config.default_val / 100:.2f}")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(config.min_val, config.max_val)
        slider.setValue(config.default_val)
        slider.valueChanged.connect(partial(self._on_slider_changed, config.name))
        self.slider_widgets[config.name] = (slider, label)

        self._layout.addWidget(slider, row, 0)
        self._layout.addWidget(label, row, 1)

    def _on_slider_changed(self, param_name: str, value: int) -> None:
        """Write the scaled slider value to the simulation and update the label."""
        setattr(self.animatedLabel.gol, param_name, value / 100)
        self.slider_widgets[param_name][1].setText(f"{param_name}: {value / 100:.2f}")

    def _reset_simulation(self) -> None:
        """Reset parameter sliders, mode, and grid to their defaults."""
        for config in SLIDER_CONFIGS:
            slider, label = self.slider_widgets[config.name]
            slider.blockSignals(True)
            slider.setValue(config.default_val)
            slider.blockSignals(False)
            setattr(self.animatedLabel.gol, config.name, config.default_val / 100)
            label.setText(f"{config.name}: {config.default_val / 100:.2f}")
        self.mode_selector.setCurrentIndex(0)
        self.animatedLabel.gol.generate_initial_state()
        self.animatedLabel.update_animation()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWidget = AnimationWidget(800, 800)
    mainWidget.show()
    sys.exit(app.exec())
