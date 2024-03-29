import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import scipy


class GameOfLife:

    def __init__(
        self,
        square_size: int = 5,
        target_fps: int = 160,
        colormap: str = "magma",
        n_intermediate_time_steps: int = 1,
        n_intermediate_alive_steps: int = 1,
        random_state: int | None = None,
        b1: float = 0.1875,
        b2: float = 0.4375,
        d1: float = 0.3125,
        d2: float = 0.4375,
        alpha_m: float = 0.03,
        alpha_n: float = 0.15,
        density: float = 0.5,
        k: float = 10,
    ) -> None:
        """
        Initialize the GameOfLife object.

        Parameters:
        square_size (int): The size of each square in the array.
        target_fps (int): The target frames per second for the game.
        colormap (str): The colormap to use for displaying the game.
        n_intermediate_steps (int): The number of intermediate steps between each full step of the game.
        random_state (int or None): The random seed for generating the initial game state. If None, a random seed is used.
        """
        self.square_size = square_size
        self.target_fps = target_fps
        self.screen_width = 1000
        self.screen_height = 1000
        self.colormap = colormap
        self.density = density
        self.dt = 1 / n_intermediate_time_steps
        self.n_intermediate_time_steps = n_intermediate_time_steps
        self.n_intermediate_alive_steps = n_intermediate_alive_steps
        self.n_squares_width = self.screen_width // self.square_size
        self.n_squares_height = self.screen_height // self.square_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.b1 = b1
        self.b2 = b2
        self.d1 = d1
        self.d2 = d2
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n
        self.random_state = random_state
        self.initialize_map()
        self.k = 4 / k

    def run_game(self) -> None:
        """
        Run the game loop.
        """
        while self.running:
            self.handle_events()
            self.calculate_next_step()
            self.display_array()
            self.clock.tick(self.target_fps)

        pygame.quit()

    def resize_image(self, array: np.ndarray) -> np.ndarray:
        """Resize and prepare the given array for display."""
        resized_image = cv2.resize(
            (array.astype(np.uint8)),
            (
                self.n_squares_width * self.square_size,
                self.n_squares_height * self.square_size,
            ),
            interpolation=cv2.INTER_LANCZOS4,
        )
        return resized_image

    def handle_events(self) -> None:
        """
        Handle user events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def display_array(self) -> None:
        """
        Display the array on the Pygame window.
        """
        array_to_show = np.clip(self.array, 0, 1) * 255
        resized_array = self.resize_image(array_to_show)
        rgb_array = self.apply_colormap(resized_array)
        surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def apply_colormap(self, resized_array) -> np.ndarray:
        """Convert values of each pixel to a color according to the selected
        matplotlib colormap."""
        colormap = plt.get_cmap(self.colormap)
        rgb_array = colormap(resized_array)[:, :, :3] * 255
        return rgb_array

    def initialize_map(self) -> None:
        """
        Initialize a binary array with random values.
        """
        rng = np.random.default_rng(self.random_state)
        array = rng.random(size=(self.n_squares_width, self.n_squares_height))
        mask = rng.random(size=array.shape) < self.density
        array = np.where(mask, array, 0)
        self.array = array

    def calculate_cell_changes(
        self, current_state, survival_conditions, birth_conditions
    ):
        growing_cells = current_state * survival_conditions
        born_cells = (1 - current_state) * birth_conditions

        return (growing_cells + born_cells).clip(0, 1)

    def calculate_next_step(self) -> None:
        """
        Apply the rules of Conway's Game of Life to update the given 2D array.
        """
        current_state = self.array
        current_state = self.sigma(k=6, x=self.array, offset=0.5)
        neighbor_sums = self.calculate_neighbor_sum(current_state).clip(0, 1)
        survival_conditions = self.calculate_transition_intervals(
            x=neighbor_sums,
            lower_threshold=self.b1,
            upper_threshold=self.b2,
        )
        birth_conditions = self.calculate_transition_intervals(
            x=neighbor_sums,
            lower_threshold=self.d1,
            upper_threshold=self.d2,
        )

        next_full_step = self.calculate_cell_changes(
            current_state, survival_conditions, birth_conditions
        )

        dx = 2 * next_full_step - 1
        dx = next_full_step - self.array
        next_intermediate_step = self.array + self.dt * dx
        self.array = next_intermediate_step.clip(0, 1)

    def calculate_neighbor_sum(self, current_state) -> np.ndarray:
        size = 20
        cell_radius = 1
        outer_radius = 3 * cell_radius
        inner_kernel = self.gaussian_kernel(sigma=cell_radius, size=size)
        outer_kernel = self.gaussian_kernel(sigma=outer_radius, size=size)
        neighborhood_kernel = outer_kernel - 1 / 9 * inner_kernel

        # outer_kernel = np.ones(shape=(3, 3))
        # inner_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        # neighborhood_kernel = (outer_kernel - inner_kernel) / 9
        neighbor_sum = scipy.ndimage.convolve(
            current_state, neighborhood_kernel, mode="wrap"
        )

        return neighbor_sum

    def calculate_transition_intervals(
        self, x: np.ndarray, lower_threshold: float, upper_threshold: float
    ) -> np.ndarray:
        sigmoid_up = self.sigma(self.k, x, lower_threshold)
        sigmoid_down = self.sigma(self.k, x, upper_threshold)
        return sigmoid_up - sigmoid_down

    def gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generates a 2D Gaussian kernel."""
        size = int(size) // 2
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()

    def sigma(self, k: float, x: np.ndarray, offset: float) -> np.ndarray:
        result = 1 / (1 + np.exp(-k * (x - offset)))
        return result


def run():
    game = GameOfLife(
        square_size=5,
        target_fps=10,
        n_intermediate_time_steps=1,
        random_state=32,
        k=0.1,
        density=0.5,
    )
    game.run_game()


run()
