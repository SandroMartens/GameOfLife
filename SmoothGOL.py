import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import scipy


class GameOfLife:
    def __init__(
        self,
        square_size: int = 5,
        target_fps: int = 60,
        colormap: str = "magma",
        n_intermediate_steps: int = 10,
        random_state: int | None = None,
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
        self.n_intermediate_steps = n_intermediate_steps
        self.n_squares_width = self.screen_width // self.square_size
        self.n_squares_height = self.screen_height // self.square_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.random_state = random_state
        self.initialize_map()
        self.kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def run_game(self) -> None:
        """
        Run the game loop.
        """
        while self.running:
            self.handle_events()
            for i in range(self.n_intermediate_steps):
                self.calculate_next_step()
                # if i % 10 == 0:
                # pass
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
        rgb_array = self.create_rgb_array(resized_array)
        surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def create_rgb_array(self, resized_array) -> np.ndarray:
        """Convert values of each pixel to a color according to the select matplotlib colormap."""
        colormap = plt.get_cmap(self.colormap)
        rgb_array = colormap(resized_array)[:, :, :3] * 255
        return rgb_array

    def initialize_map(self) -> None:
        """
        Initialize a binary array with random values.
        """
        rng = np.random.default_rng(self.random_state)
        # array = rng.integers(2, size=(self.n_squares_width, self.n_squares_height))
        array = rng.random(size=(self.n_squares_width, self.n_squares_height))
        mask = rng.random(size=array.shape) < 0.5
        array = np.where(mask, array, 0)
        self.array = array

    def calculate_next_step(self) -> None:
        """
        Apply the rules of Conway's Game of Life to update the given 2D array.
        """
        factor = 1 / self.n_intermediate_steps
        current_array = self.array
        neighbor_sums = self.calculate_neighbor_sum()
        survival_conditions = np.logical_and(neighbor_sums >= 1.5, neighbor_sums <= 3.5)
        birth_conditions = np.logical_and(neighbor_sums >= 1.9, neighbor_sums <= 3)

        growing_cells = current_array * survival_conditions
        shrinking_cells = current_array * ~survival_conditions
        born_cells = (1 - current_array) * birth_conditions
        dying_cells = (1 - current_array) * (~birth_conditions)

        next_full_step = growing_cells + born_cells - dying_cells - shrinking_cells
        # next_full_step = np.clip(next_full_step, -1, 1)

        next_intermediate_step = current_array + factor * next_full_step
        # next_intermediate_step = (1 - factor) * current_array + factor * next_full_step
        next_intermediate_step = np.clip(next_intermediate_step, -0.4, 1.5)

        self.array = next_intermediate_step

    def calculate_neighbor_sum(self) -> np.ndarray:
        inner_kernel = self.gaussian_kernel(sigma=0.5, size=7)
        outer_kernel = self.gaussian_kernel(sigma=1.5, size=7)
        outer_kernel *= 9

        # outer_kernel = np.ones(shape=(3, 3))
        # inner_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        neighborhood_kernel = outer_kernel - inner_kernel
        neighbor_sum = scipy.ndimage.convolve(
            self.array, neighborhood_kernel, mode="wrap"
        )

        return neighbor_sum

    def gaussian_kernel(self, size=10, sigma=3) -> np.ndarray:
        """Generates a 2D Gaussian kernel."""
        size = int(size) // 2
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()


def run():
    game = GameOfLife(
        square_size=5, n_intermediate_steps=10, random_state=32, colormap="magma"
    )
    game.run_game()


run()
