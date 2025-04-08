from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from marching_squares import contour_grid_to_path_list, corners_to_squares, squares_to_contour_grid

IMAGE_PATH = "./images/sheapland tiles on full size map.png"
EMPTY_PIXEL = [255, 255, 255]  # RGB value representing an empty pixel


def load_image(img_path: str) -> np.ndarray:
    """
    Load the image and convert it to a numpy array.
    """
    img = Image.open(img_path).convert("RGB")
    pixel_data = np.array(img)
    return pixel_data


def find_unique_colours(pixel_data: np.ndarray) -> set[tuple[int, int, int]]:
    """
    Find unique colours in an image stored as a numpy array.
    The colour that matches the EMPTY_PIXEL is ignored.

    Return a set of the colours found.
    """
    # Reshape image into (height * width, 3) array
    flat_pixels = pixel_data.reshape(-1, 3)

    # Remove pixels that match EMPTY_PIXEL
    mask = ~np.all(flat_pixels == EMPTY_PIXEL, axis=1)
    filtered_pixels = flat_pixels[mask]

    # Get unique colours as tuples
    unique = np.unique(filtered_pixels, axis=0)
    return set(map(tuple, unique))  # type: ignore


def mask_colour(pixel_data: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """
    Return a numpy array with the same dimensions of the input image,
    but with only one channel, which is either True or False, depending
    on whether the pixel there matched the colour passed in.
    """

    return np.all(pixel_data == colour, axis=-1)


if __name__ == "__main__":
    pixel_data = load_image(IMAGE_PATH)
    print("loaded image")

    unique_colours = find_unique_colours(pixel_data)
    print(f"Unique region colours found: {unique_colours}")

    masked = mask_colour(pixel_data, unique_colours.pop())
    print("got mask")

    # plt.imshow(masked, cmap="gray")
    # plt.title("Masked Region")
    # plt.show()

    # plt.imshow(pixel_data)
    # plt.show()

    squares = corners_to_squares(masked)
    print("converted to squares")
    contour_lines = squares_to_contour_grid(squares)
    print("converted to contours")
    pen_paths = contour_grid_to_path_list(contour_lines)
    print("converted to paths")

    print([[[point[0], point[1]] for point in path] for path in pen_paths])
