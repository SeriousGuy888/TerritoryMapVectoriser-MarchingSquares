from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

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
            nw, ne, sw, se = corners[y][x], corners[y][x + 1], corners[y + 1][x], corners[y + 1][x + 1]

            # Encode the state of the four corners as a single number.
            state = nw << 3 | ne << 2 | se << 1 | sw
            array[y][x] = state

    return array


def squares_to_contour_grid(squares: np.ndarray) -> np.ndarray:
    """
    Given a 2d array of squares, where each cell is a number from 0 to 15 describing the state of the square,
    return a new 2d array with all the square states converted to the appropriate contour lines at that square.

    Using lookup table from https://en.wikipedia.org/wiki/Marching_squares#Basic_algorithm

    >>> contour_grid = squares_to_contour_grid(np.array([[10]]))
    >>> n = contour_grid[0][0].item()
    >>> n == 9
    True
    >>> l = ContourLines.from_number(n)
    >>> l.n_e
    True
    >>> l.s_w
    True
    """

    height, width = squares.shape

    contour_grid = np.empty((height, width), dtype=np.int8)

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

            contour_grid[y][x] = contour.to_number()

    return contour_grid


def contour_grid_to_path_list(contour_grid: np.ndarray, coordinate_offset: tuple[int, int]) -> \
        list[list[tuple[float, float]]]:
    """
    Given a grid of numbers that can be deserialised to ContourLine,
    return a list of pen paths.

    The pen paths will be offset by the coordinate_offset given in (x, y) format.
    """
    offset_x, offset_y = coordinate_offset

    # Line segments that have to be included in the SVG path.
    segments = []

    places_with_contour_lines = np.argwhere(contour_grid != 0)
    for y, x in places_with_contour_lines:
        contour_num = contour_grid[y][x]
        contour = ContourLines.from_number(contour_num)
        segments.extend(contour.get_line_segments(float(x) + offset_x, float(y) + offset_y))

    paths = build_paths_from_segment_list(segments)
    for p in paths:
        optimise_path(p)

    return paths


def pen_paths_to_svg_path(pen_paths: list[list[tuple[float, float]]]):
    """
    Given a list of pen paths -- a list of coordinates -- return an SVG path element.
    """

    # The pen movements of the SVG path
    d_attribute = ""

    for path in pen_paths:
        first_point = path[0]
        d_attribute += f"M {first_point[1]} {first_point[0]} "

        for i in range(1, len(path)):
            point = path[i]
            d_attribute += f"L {point[1]} {point[0]} "

    return f"<path d=\"{d_attribute}\" />"


def build_paths_from_segment_list(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> \
        list[list[tuple[float, float]]]:
    """
    Given a list of line segments, join segments together wherever two segments share an endpoint, and
    return a list of polylines.

    >>> build_paths_from_segment_list([
    ...     ((0, 0), (1, 1)),
    ...     ((1, 1), (2, 2)),
    ...     ((10, 10), (20, 20)),
    ... ])
    [[(0, 0), (1, 1), (2, 2)], [(10, 10), (20, 20)]]
    """
    segment_neighbourships = defaultdict(list)

    for point_a, point_b in segments:
        segment_neighbourships[point_a].append(point_b)
        segment_neighbourships[point_b].append(point_a)

    visited = set()
    paths = []

    for starting_point in segment_neighbourships:
        if starting_point in visited:
            continue

        path = [starting_point]
        current_point = starting_point

        visited.add(current_point)

        while True:
            neighbours = [p for p in segment_neighbourships[current_point] if p not in visited]  # type: list
            if not neighbours:
                break

            next_point = neighbours[0]
            path.append(next_point)
            visited.add(next_point)
            current_point = next_point

        paths.append(path)

    return paths


def optimise_path(path: list[tuple[float, float]]) -> None:
    """
    Given a path -- a list of coordinate pairs -- remove any redundant points in the path (MUTATES INPUT).
    A point is "redundant" if the segment preceding it and the segment following it move in the same direction.

    Preconditions:
        - len(path) >= 1

    >>> path = [(0, 0), (1, 1), (5, 5), (6, 6), (10, 6)]
    >>> optimise_path(path)
    >>> path
    [(0, 0), (6, 6), (10, 6)]
    """

    i = 1
    while i < len(path) - 1:
        prev_point = path[i - 1]
        curr_point = path[i]
        next_point = path[i + 1]

        dy_1 = (curr_point[0] - prev_point[0])
        dx_1 = curr_point[1] - prev_point[1]
        dx_2 = next_point[1] - curr_point[1]
        dy_2 = (next_point[0] - curr_point[0])

        slope_1 = dy_1 / dx_1 if dx_1 != 0 else float("inf") * dy_1
        slope_2 = dy_2 / dx_2 if dx_2 != 0 else float("inf") * dy_2

        if math.isclose(slope_1, slope_2, abs_tol=1e-4):
            path.pop(i)
        else:
            i += 1


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

    def get_line_segments(self, x: float, y: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Given the x and y representing the top left corner of the square at which these contour lines are placed,
        return a list of the line segments that these contour lines define.
        The line segments are a tuple of the coordinates of their two endpoints.
        """

        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

        if self.n_s:
            segments.append(((x + 0.5, y), (x + 0.5, y + 1)))
        elif self.e_w:
            segments.append(((x, y + 0.5), (x + 1, y + 0.5)))
        elif self.n_w or self.s_e:
            if self.n_w:
                segments.append(((x + 0.5, y), (x, y + 0.5)))
            if self.s_e:
                segments.append(((x + 0.5, y + 1), (x + 1, y + 0.5)))
            return segments
        elif self.n_e or self.s_w:
            if self.n_e:
                segments.append(((x + 0.5, y), (x + 1, y + 0.5)))
            if self.s_w:
                segments.append(((x + 0.5, y + 1), (x, y + 0.5)))

        return segments

    def to_number(self) -> int:
        """
        Return an integer that represents this contour line state.
        """
        return self.n_s << 5 | self.e_w << 4 | self.n_e << 3 | self.n_w << 2 | self.s_e << 1 | self.s_w

    @staticmethod
    def from_number(number: int) -> ContourLines:
        """
        Return a contour line state based on the number.
        """
        return ContourLines(
            n_s=bool(number & 1 << 5),
            e_w=bool(number & 1 << 4),
            n_e=bool(number & 1 << 3),
            n_w=bool(number & 1 << 2),
            s_e=bool(number & 1 << 1),
            s_w=bool(number & 1 << 0),
        )
