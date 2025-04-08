import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from marching_squares import contour_grid_to_path_list, corners_to_squares, squares_to_contour_grid

IMAGE_PATH = "./images/sheapland tiles on full size map.png"
EMPTY_PIXEL = [255, 255, 255]  # RGB value representing an empty pixel
ORIGIN_OFFSET = [5001, 5001]  # the position in the image to consider as (0, 0) when outputting paths.


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


def mask_colour(pixel_data: np.ndarray, colour: tuple[int, int, int]) -> \
        tuple[np.ndarray, tuple[int, int]]:
    """
    Return two things:

    1. a 2d numpy array which holds in each cell either True or False, depending
       on whether the pixel there matched the colour passed in.
       The returned array is cropped to be the smallest rectangle possible that contains
       all the matching pixels.

    2. a crop offset, representing what position (0, 0) in the cropped (returned) numpy array
       actually represents in the original image space.
       The format is (x, y)
    """
    height, width, _ = pixel_data.shape

    # Full mask in the original image space.
    # Might be too big to handle in later functions, so needs to be cropped
    full_mask = np.all(pixel_data == colour, axis=-1)

    # Figure out how much we can crop without removing any True cells
    y_values, x_values = np.where(full_mask)

    if len(y_values) == 0 or len(x_values) == 0:
        return np.zeros((0, 0)), (0, 0)

    # The bounds of the crop rectangle
    # Padded by 1 pixel on each side because we run Marching Squares on each four-way pixel intersection.
    # Since we need the corners and edges of the True area to be processed, we need a pixel of padding if possible.
    min_y = max(0, y_values.min() - 1)
    max_y = min(height, y_values.max() + 1)
    min_x = max(0, x_values.min() - 1)
    max_x = min(width, x_values.max() + 1)

    return full_mask[min_y:max_y + 1, min_x:max_x + 1], (min_x, min_y)


def get_paths_for_colour(pixel_data: np.ndarray, colour: tuple[int, int, int]):
    """
    Given the original image pixel data and the colour to look for, isolate that colour and
    return a list of pen paths that trace the regions occupied by that exact colour.
    """

    masked, mask_offset = mask_colour(pixel_data, colour)
    # print(f"got mask for {colour}")

    squares = corners_to_squares(masked)
    # print("converted to squares")
    contour_lines = squares_to_contour_grid(squares)
    # print("converted to contours")
    pen_paths = contour_grid_to_path_list(contour_lines, mask_offset)
    # print("converted to paths")

    # apply the origin offset to the paths
    # also change all the Tuple coordinates to Lists
    #   (because im going to paste this into a JSON file)
    modified_paths = [[[point[0] - ORIGIN_OFFSET[0], point[1] - ORIGIN_OFFSET[1]] for point in path] for
                      path in pen_paths]

    return modified_paths


if __name__ == "__main__":
    pixel_data = load_image(IMAGE_PATH)
    print("loaded image")

    unique_colours = find_unique_colours(pixel_data)
    print(f"Unique region colours found: {unique_colours}")

    # plt.imshow(masked, cmap="gray")
    # plt.title("Masked Region")
    # plt.show()

    # plt.imshow(pixel_data)
    # plt.show()

    tiles = {}
    for colour in tqdm(unique_colours):
        paths = get_paths_for_colour(pixel_data, colour)

        r, g, b = colour
        colour_hex_code = f"{r:0>2x}{g:0>2x}{b:0>2x}"

        tiles[colour_hex_code] = paths

    with open("tiles.json", "w") as f:
        json.dump(tiles, f, indent=2)
