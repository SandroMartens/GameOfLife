import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


# Button class
class Button:
    def __init__(self, color, x, y, width, height, text=""):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self, screen, outline=None):
        if outline:
            pygame.draw.rect(
                screen,
                outline,
                (self.x - 2, self.y - 2, self.width + 4, self.height + 4),
                0,
            )

        pygame.draw.rect(
            screen, self.color, (self.x, self.y, self.width, self.height), 0
        )

        if self.text != "":
            font = pygame.font.SysFont("comicsans", 30)
            text = font.render(self.text, 1, BLACK)
            screen.blit(
                text,
                (
                    self.x + (self.width / 2 - text.get_width() / 2),
                    self.y + (self.height / 2 - text.get_height() / 2),
                ),
            )

    def is_over(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if (
            self.x < pos[0] < self.x + self.width
            and self.y < pos[1] < self.y + self.height
        ):
            return True
        return False


# Main loop
def main():
    button = Button(GREEN, 150, 100, 250, 100, "Change Value")
    parameter_value = 0

    running = True
    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if button.is_over(pos):
                    parameter_value += 1  # Change parameter value

        button.draw(screen, BLACK)

        # Display the parameter value
        font = pygame.font.SysFont("comicsans", 30)
        value_text = font.render("Value: " + str(parameter_value), 1, BLACK)
        screen.blit(value_text, (220, 250))

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
