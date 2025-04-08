from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from marching_squares import contour_grid_to_path_list, corners_to_squares, squares_to_contour_grid

IMAGE_PATH = "./images/sheapland tiles.png"
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
    unique_colours = set()
    for row in pixel_data:
        for pixel in row:
            if not np.array_equal(pixel, EMPTY_PIXEL):
                unique_colours.add(tuple(pixel))
    return unique_colours  # type: ignore


def mask_colour(pixel_data: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """
    Return a numpy array with the same dimensions of the input image,
    but with only one channel, which is either True or False, depending
    on whether the pixel there matched the colour passed in.
    """

    return np.all(pixel_data == colour, axis=-1)


if __name__ == "__main__":
    pixel_data = load_image(IMAGE_PATH)

    unique_colours = find_unique_colours(pixel_data)
    # print(f"Unique region colours found: {unique_colours}")

    masked = mask_colour(pixel_data, unique_colours.pop())

    # plt.imshow(masked, cmap="gray")
    # plt.title("Masked Region")
    # plt.show()

    # plt.imshow(pixel_data)
    # plt.show()

    squares = corners_to_squares(masked)
    contour_lines = squares_to_contour_grid(squares)
    pen_paths = contour_grid_to_path_list(contour_lines)

    print([[[point[0], point[1]] for point in path] for path in pen_paths])
