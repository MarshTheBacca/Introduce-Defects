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
    angle = np.arccos(dot_product_over_magnitudes)
    angle = np.degrees(angle)
    # Return the acute angle
    return min(angle, 180 - angle)


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
