import numpy as np


def settify(iterable: list) -> list:
    """
    Used to remove duplicates from a list while preserving order. This is for lists with mutable types, ie, are not hashable.
    """
    return_list = []
    for item in iterable:
        if item not in return_list:
            return_list.append(item)
    return return_list


def find_common_elements(lists: list[list]) -> list:
    """
    Return a list of elements that are common to all lists.
    """
    first_list = lists[0]
    common_elements = [element for element in first_list if all(
        element in lst for lst in lists[1:])]
    return common_elements


def rounded_even_sqrt(number: float) -> int:
    """
    Return the square root of a number rounded down to the nearest even number.
    """
    if number < 0:
        raise ValueError("Cannot take the square root of a negative number.")
    rounded_root = np.floor(np.sqrt(number))
    if rounded_root % 2 == 0:
        return int(rounded_root)
    return int(rounded_root - 1)


def pbc_vector(vector1: np.ndarray, vector2: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    Calculate the vector difference between two vectors, taking into account periodic boundary conditions.
    Remember dimensions = [[xlo, ylo], [xhi, yhi]]
    """
    if len(vector1) != len(vector2) or len(vector1) != len(dimensions):
        raise ValueError("Vectors must have the same number of dimensions.")
    difference_vector = np.subtract(vector2, vector1)
    dimension_ranges = dimensions[1] - dimensions[0]
    half_dimension_ranges = dimension_ranges / 2
    difference_vector = (difference_vector + half_dimension_ranges) % dimension_ranges - half_dimension_ranges
    return difference_vector


def calculate_angle(coord_1: np.ndarray, coord_2: np.ndarray, coord_3: np.ndarray) -> float:
    """
    Returns the angle between three points in degrees.
    """
    vector1 = np.array(coord_1) - np.array(coord_2)
    vector2 = np.array(coord_3) - np.array(coord_2)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    # Clamp the value to the range [-1, 1] to avoid RuntimeWarning from floating point errors
    dot_product_over_magnitudes = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    return np.degrees(np.arccos(dot_product_over_magnitudes))


def arrange_clockwise(center_coord: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Sorts the coordinates in a clockwise direction around the center coordinate.

    Args:
        center_coord: np.ndarray, the center coordinate
        coords: np.ndarray, the coordinates to be sorted

    Returns:
        np.ndarray, the sorted coordinates
    """
    # Calculate the angles of the vectors from the center coordinate to each coordinate
    angles = np.arctan2(coords[:, 1] - center_coord[1], coords[:, 0] - center_coord[0])
    angles = angles % (2 * np.pi)  # adjust the angles to be between 0 and 2*pi
    # Sort the coordinates based on the angles
    sorted_coords = coords[np.argsort(angles)]
    return sorted_coords


def calculate_angles_around_node(center_coord: np.ndarray, neighbour_coords: np.ndarray) -> np.ndarray:
    """
    Calculates the angles (in degrees) between neighbours around a center coordinate in a clockwise direction.

    Args:
        center_coord: np.ndarray, the center coordinate
        neighbour_coords: np.ndarray, the coordinates of the neighbours

    Returns:
        np.ndarray, the angles between neighbours around the center coordinate in a clockwise direction
    """
    # Sort the neighbour coordinates in a clockwise direction around the node
    neighbour_coords = arrange_clockwise(center_coord, neighbour_coords)
    # Calculate the angles of the vectors from the node to each neighbour
    angles = np.arctan2(neighbour_coords[:, 1] - center_coord[1], neighbour_coords[:, 0] - center_coord[0])
    angles = angles % (2 * np.pi)  # adjust the angles to be between 0 and 2*pi
    # Calculate the differences between each pair of consecutive angles in a clockwise direction
    angles = np.concatenate([angles, [angles[0]]])  # append the first angle to handle the wrap-around at 2*pi
    angles[:-1] = np.diff(angles)  # calculate the differences
    angles = angles[:-1] % (2 * np.pi)  # use modulo 2*pi to handle the wrap-around at 2*pi
    angles = np.degrees(angles)  # convert to degrees
    return angles


def is_pbc_bond(coord_1: np.ndarray, coord_2: np.ndarray, dimensions: np.ndarray) -> bool:
    """
    Identifies bonds that cross the periodic boundary. If the length of the bond is 
    more than 10% longer than the distance between the two nodes with periodic boundary conditions,
    then it is considered a periodic bond.

    Args:
        coord_1: np.ndarray, coordinates of the first node
        coord_2: np.ndarray, coordinates of the second node

    Returns:
        bool, True if the bond is periodic, False otherwise
    """
    if np.linalg.norm(coord_1 - coord_2) > np.linalg.norm(pbc_vector(coord_1, coord_2, dimensions)) * 1.1:
        return True
    return False
