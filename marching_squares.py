import numpy as np


def corners_to_squares(corners: np.ndarray) -> np.ndarray:
    """
    Given a 2d array of booleans, where each cell represents a corner of a Marching Squares square,
    return a 2d array, where each cell is a number representing the "state" of that Marching Squares square.

    The "state" represents the corners of the square as a number from 0 to 15, where the
        - 2^3 bit represents the Northwest corner
        - 2^2 bit represents the Northeast corner
        - 2^1 bit represents the Southeast corner
        - 2^0 bit represents the Southwest corner

    Preconditions:
        - The mask is a 2d array of booleans.
        - The mask's dimensions are at least 2x2.

    Postconditions:
        - The returned array's width and height will be one less than the width and height of the input array.

    >>> mask_ = np.array([
    ...     [True, False],
    ...     [False, True],
    ... ])
    >>> squares = corners_to_squares(mask_)
    >>> squares[0][0] == 8 + 0 + 2 + 0
    True
    """

    height, width = corners.shape

    # Each "square" that the marching squares algorithm goes through is actually
    # the intersection between four pixels in the image mask provided.
    #
    # So we actually need one less than the width and height of the mask,
    # since each pixel in the mask is actually a corner of a square.

    array = np.empty((height - 1, width - 1), dtype=np.int8)

    for y in range(height - 1):
        for x in range(width - 1):
            # Get the four corners.
            nw, ne, sw, se = corners[x][y], corners[x + 1][y], corners[x][y + 1], corners[x + 1][y + 1]

            # Encode the state of the four corners as a single number.
            state = nw << 3 | ne << 2 | se << 1 | sw
            array[y][x] = state

    return array
