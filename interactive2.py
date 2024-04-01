import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

# Example numpy array (replace this with your actual pixel data)
pixel_data = np.random.rand(10, 10)  # 10x10 array of random values


def main():
    def update_image(brightness=1.0, contrast=1.0):
        # Adjust the pixel data based on brightness and contrast sliders
        # This is a simple example; you might need a more complex function
        adjusted_image = np.clip(pixel_data * brightness * contrast, 0, 1)

        # Display the image
        plt.imshow(adjusted_image, cmap="gray")
        plt.axis("off")  # Hide axis
        plt.show()

    # Create interactive UI with sliders for brightness and contrast
    interact(
        update_image,
        brightness=widgets.FloatSlider(
            value=1.0, min=0, max=2, step=0.1, description="Brightness:"
        ),
        contrast=widgets.FloatSlider(
            value=1.0, min=0, max=2, step=0.1, description="Contrast:"
        ),
    )


if __name__ == "__main__":
    main()
