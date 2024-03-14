import cv2
import numpy as np
import pygame
import scipy
import scipy.signal


class GameOfLife:
    def __init__(
        self,
        square_size: int = 5,
        ruleset: tuple[list[int], list[int]] = ([2, 3], [3]),
        target_fps: int = 20,
        random_state=None,
    ) -> None:
        """
        Initialize the GameOfLife object.

        Parameters:
        square_size (int): The size of each square in the array.
        """
        self.square_size = square_size
        self.target_fps = target_fps
        self.screen_width = 1000
        self.screen_height = 1000
        self.ruleset = ruleset
        self.n_squares_width = self.screen_width // self.square_size
        self.n_squares_height = self.screen_height // self.square_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.random_state = random_state
        self.initialize_map()

    def run_game(self) -> None:
        """
        Run the game loop.
        """
        while self.running:
            self.handle_events()
            self.calculate_next_generation()
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

        Parameters:
        array (np.ndarray): The array to display.
        """
        resized_array = self.resize_image(self.array)
        surface = pygame.surfarray.make_surface(resized_array * 255)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def initialize_map(self) -> None:
        """
        Initialize a binary array with random values.
        """
        rng = np.random.default_rng(self.random_state)
        array = rng.integers(2, size=(self.n_squares_width, self.n_squares_height))
        self.array = array

    def calculate_next_generation(self) -> None:
        """
        Apply the rules of Conway's Game of Life to update the given 2D array.
        """
        set_alive_to_survive, set_dead_to_alive = self.ruleset
        array = self.array
        neighbor_sum = self.calculate_neighbor_sum(array)

        # Cells that will survive
        survival_conditions = np.isin(neighbor_sum, set_alive_to_survive) & (array == 1)
        # Cells that will come to life
        birth_conditions = np.isin(neighbor_sum, set_dead_to_alive) & (array == 0)

        # Cells that do not meet the survival conditions and are currently alive will die
        new_array = np.where(~survival_conditions & (array == 1), 0, array)
        new_array = np.where(birth_conditions, 1, new_array)  # Apply birth conditions

        self.array = new_array

    def calculate_neighbor_sum(self, array: np.ndarray) -> np.ndarray:
        """
        Calculate the sum of neighbors for each cell in the given 2D array.
        """
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_sum = scipy.signal.convolve2d(
            array, kernel, mode="same", boundary="wrap"
        )
        return neighbor_sum


def run():
    game = GameOfLife(square_size=5, ruleset=([2, 3], [3]), random_state=32)
    game.run_game()


run()
