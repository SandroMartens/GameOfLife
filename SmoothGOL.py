import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import scipy
import scipy.ndimage


class SmoothGameOfLife:
    """The GameOfLife class represents a simulation of Conway's Game of Life."""

    def __init__(
        self,
        target_fps: int = 10,
        colormap: str = "magma",
        dt: float = 1,
        random_state: int | None = None,
        survival_lower_threshold: float = 0.1875,
        survival_upper_threshold: float = 0.4375,
        birth_lower_threshold: float = 0.3125,
        birth_upper_threshold: float = 0.4375,
        screen_width: int = 1000,
        screen_height: int = 1000,
        field_width: int = 200,
        field_height: int = 200,
        alpha_m: float = 0.15,
        cell_size: float = 1,
        init_density: float = 0.5,
        mode: int = 1,
        k: float = 0.18,
    ) -> None:
        """Initialize the GameOfLife object with specified configurations.

        Parameters:
        - square_size (int): The size of each square in the grid, affecting the resolution of the simulation.
        - target_fps (int): The target frames per second for the game, controlling the speed of the simulation.
        - colormap (str): The name of the matplotlib colormap used for displaying the game state.
        - n_intermediate_time_steps (int): The number of intermediate steps to compute between each displayed frame, for smoother transitions.
        - random_state (int | None): Seed for the random number generator for reproducible initial states. If None, a random seed is used.
        - b1 (float): Lower threshold for the birth condition in the transition function.
        - b2 (float): Upper threshold for the birth condition in the transition function.
        - d1 (float): Lower threshold for the survival condition in the transition function.
        - d2 (float): Upper threshold for the survival condition in the transition function.
        - alpha_m (float): Parameter `alpha_m` influencing the model (not directly used in the provided code snippet).
        - alpha_n (float): Parameter `alpha_n` influencing the model (not directly used in the provided code snippet).
        - init_density (float): The initial density of alive cells in the game grid.
        - k (float): Parameter influencing the steepness of the transition function between states.

        Initializes the game window, internal state, and starts with a randomly generated map based on the given parameters.
        """
        self.target_fps = target_fps
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.field_width = field_width
        self.field_height = field_height
        self.colormap = colormap
        self.init_density = init_density
        self.dt = dt
        self.n_intermediate_time_steps = dt
        self.running = True
        self.birth_lower_threshold = birth_lower_threshold
        self.birth_upper_threshold = birth_upper_threshold
        self.survival_lower_threshold = survival_lower_threshold
        self.survival_upper_threshold = survival_upper_threshold
        self.cell_size = cell_size
        self.alpha_m = alpha_m
        self.mode = mode
        # self.alpha_n = alpha_n
        self.random_state = random_state
        self.k = k
        self.k_slider = Slider(50, 950, 200, 20, 0.01, 1.0, self.k, "k")

    def run_game(self) -> None:
        """
        Run the game loop.
        """
        self.generate_initial_state()
        try:
            while self.running:
                self.handle_events()
                self.calculate_next_step()
                self.display_array()
                self.clock.tick(self.target_fps)
            pygame.quit()

        finally:
            pygame.quit()

    def resize_image(self, array: np.ndarray) -> np.ndarray:
        """Resize and prepare the given array for display."""
        resized_image = cv2.resize(
            (array.astype(np.uint8)),
            (self.screen_width, self.screen_height),
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # If 'R' key is pressed
                    self.restart_simulation()
            self.k_slider.handle_event(event)

    def restart_simulation(self) -> None:
        """
        Restart the simulation by reinitializing the game state.
        """
        self.generate_initial_state()  # Regenerate the initial state
        self.running = True

    def display_array(self) -> None:
        """
        Display the array on the Pygame window.
        """
        array_to_show = np.clip(self.current_state_array, 0, 1) * 255
        resized_array = self.resize_image(array_to_show)
        rgb_array = self.apply_colormap(resized_array)
        surface = pygame.surfarray.make_surface(rgb_array)
        self.screen.blit(surface, (0, 0))
        self.k_slider.draw(self.screen)
        self.k = self.k_slider.val

        pygame.display.flip()

    def apply_colormap(self, resized_array: np.ndarray) -> np.ndarray:
        """Convert values of each pixel to a color according to the selected
        matplotlib colormap."""
        colormap = plt.get_cmap(self.colormap)
        rgb_array = colormap(resized_array)[:, :, :3] * 255
        return rgb_array

    def generate_initial_state(self) -> None:
        """
        Generate the initial state of the game.
        """
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        rng = np.random.default_rng(self.random_state)
        array = rng.random(size=(self.field_width, self.field_height))
        mask = rng.random(size=array.shape) < self.init_density
        masked_array = np.where(mask, array, 0)
        self.current_state_array = masked_array

    def calculate_new_state(
        self,
        current_state: np.ndarray,
        survival_conditions: np.ndarray,
        birth_conditions: np.ndarray,
    ) -> np.ndarray:
        growing_cells = current_state * survival_conditions
        born_cells = (1 - current_state) * birth_conditions

        new_state = (growing_cells + born_cells).clip(0, 1)

        return new_state

    def calculate_conditions(
        self, x: np.ndarray, lower_threshold: float, upper_threshold: float
    ) -> np.ndarray:
        sigmoid_up = self.apply_sigmoid_function(self.k, x, lower_threshold)
        sigmoid_down = self.apply_sigmoid_function(self.k, x, upper_threshold)
        return sigmoid_up - sigmoid_down

    def calculate_next_step(self) -> None:
        """
        Apply the rules of Conway's Game of Life to update the given 2D array.
        """
        cell_radius = self.cell_size
        neighboorhood_region_radius = 3 * cell_radius
        kernel_diameter_cell = 5 * cell_radius
        kernel_diameter_neighborhood_region = 5 * neighboorhood_region_radius

        neighbor_sums = self.apply_convolution_to_state(
            radius=neighboorhood_region_radius,
            size=kernel_diameter_neighborhood_region,
        ).clip(0, 1)
        cell_sums = self.apply_convolution_to_state(
            radius=cell_radius,
            size=kernel_diameter_cell,
        ).clip(0, 1)

        neighbor_sums = (neighbor_sums - 1 / 9 * cell_sums) / (8 / 9)

        aliveness = self.apply_sigmoid_function(k=self.alpha_m, x=cell_sums, offset=0.5)

        birth_conditions = self.calculate_conditions(
            x=neighbor_sums,
            lower_threshold=self.birth_lower_threshold,
            upper_threshold=self.birth_upper_threshold,
        )
        survival_conditions = self.calculate_conditions(
            x=neighbor_sums,
            lower_threshold=self.survival_lower_threshold,
            upper_threshold=self.survival_upper_threshold,
        )

        next_full_step = self.calculate_new_state(
            current_state=aliveness,
            survival_conditions=survival_conditions,
            birth_conditions=birth_conditions,
        )

        dx = self.calculate_delta_state(cell_sums, next_full_step)
        next_intermediate_step = self.current_state_array + self.dt * dx
        self.current_state_array = next_intermediate_step.clip(0, 1)

    def calculate_delta_state(
        self, cell_sums: np.ndarray, next_full_step: np.ndarray
    ) -> np.ndarray:
        if self.mode == 1:
            dx = next_full_step - self.current_state_array
        elif self.mode == 2:
            dx = 2 * next_full_step - 1
        elif self.mode == 3:
            dx = next_full_step - cell_sums
        return dx

    def apply_convolution_to_state(self, radius: float, size: float) -> np.ndarray:
        kernel = self.get_gaussian_kernel(sigma=radius, size=size)
        result = scipy.ndimage.convolve(self.current_state_array, kernel, mode="wrap")
        return result

    def calculate_transition_intervals(
        self, x: np.ndarray, lower_threshold: float, upper_threshold: float
    ) -> np.ndarray:
        sigmoid_up = self.apply_sigmoid_function(self.k, x, lower_threshold)
        sigmoid_down = self.apply_sigmoid_function(self.k, x, upper_threshold)
        return sigmoid_up - sigmoid_down

    def get_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generates a 2D Gaussian kernel."""
        size = int(size) // 2
        x, y = np.mgrid[-size : size + 1, -size : size + 1]
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return g / g.sum()

    def apply_sigmoid_function(
        self, k: float, x: np.ndarray, offset: float
    ) -> np.ndarray:
        """Apply a sigmoid function with a given"""
        result = 1 / (1 + np.exp(-4 / k * (x - offset)))
        return result


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, name):
        self.rect = pygame.Rect(x, y, w, h)  # Slider position and size
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val  # Current value
        self.name = name
        self.is_held = False  # Whether the slider is being dragged

    def draw(self, screen):
        # Draw the slider track
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        # Draw the slider handle
        handle_x = (self.val - self.min_val) / (
            self.max_val - self.min_val
        ) * self.rect.width + self.rect.x
        pygame.draw.rect(
            screen, (200, 200, 200), (handle_x - 10, self.rect.y, 20, self.rect.height)
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_held = False
        elif event.type == pygame.MOUSEMOTION and self.is_held:
            # Update the slider value based on mouse position
            mouse_x, _ = event.pos
            self.val = (mouse_x - self.rect.x) / self.rect.width * (
                self.max_val - self.min_val
            ) + self.min_val
            self.val = max(min(self.val, self.max_val), self.min_val)  # Clamp value


def main():
    game = SmoothGameOfLife(
        # target_fps=50,
        # dt=0.2,
        random_state=32,
        # k=0.1,
        init_density=0.6,
        # cell_size=3,
        # b1=0.278,
        # b2=0.365,
        # d1=0.267,
        # d2=0.445,
    )
    game.run_game()


if __name__ == "__main__":
    main()
