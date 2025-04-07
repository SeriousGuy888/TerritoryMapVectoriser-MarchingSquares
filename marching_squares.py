import numpy as np
from attr import dataclass


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


def squares_to_contour_lines(squares: np.ndarray) -> np.ndarray:
    """
    Given a 2d array of squares, where each cell is a number from 0 to 15 describing the state of the square,
    return a new 2d array with all the square states converted to the appropriate contour lines at that square.

    Using lookup table from https://en.wikipedia.org/wiki/Marching_squares#Basic_algorithm
    """

    height, width = squares.shape

    contours = np.empty((height, width))

    for y in range(height):
        for x in range(width):
            state = squares[y][x]

            contour = ContourLines()
            match state:
                case 0b0001 | 0b1110:
                    contour.s_w = True
                case 0b0010 | 0b1101:
                    contour.s_e = True
                case 0b0011 | 0b1100:
                    contour.e_w = True
                case 0b0100 | 0b1011:
                    contour.n_e = True
                case 0b0101:  # saddle cases
                    contour.n_w = True
                    contour.s_e = True
                case 0b1010:
                    contour.n_e = True
                    contour.s_w = True
                case 0b0110 | 0b1001:
                    contour.n_s = True
                case 0b0111 | 0b1000:
                    contour.n_w = True

            contours[y][x] = contour  # broken for now because wrong datatype

    return contours


@dataclass
class ContourLines:
    """
    Represents the contour lines at a square, where each attribute represents whether a connection exists
    between two particular sides of the square.
    """

    n_s: bool = False
    e_w: bool = False
    n_e: bool = False
    n_w: bool = False
    s_e: bool = False
    s_w: bool = False
