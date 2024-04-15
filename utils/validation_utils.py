from typing import Optional
import sys

from typing import Callable, Optional, TypeVar

NUMBER = TypeVar('NUMBER', int, float)


class UserCancelledError(Exception):
    pass


def get_valid_num(prompt: str, convert: Callable[[str], NUMBER], lower: float | int = float("-inf"),
                  upper: float | int = float("inf"),
                  exit_string: Optional[str] = None) -> NUMBER:
    """
    Obtains a valid number from the user within a given range

    Args:
        prompt (str): The prompt to display to the user
        convert (Callable[[str], NUMBER]): The function to convert the input to the desired type
        lower (float | int): The lower bound of the range
        upper (float | int): The upper bound of the range
        exit_string (str): The string to enter to exit the prompt
    Returns:
        amswer (NUMBER): The valid input entered by the user
    Raises:
        UserCancelledError: If the user cancels the input
    """
    while True:
        answer = input(prompt)
        if answer == exit_string:
            raise UserCancelledError("User cancelled input")
        try:
            answer = convert(answer)
        except ValueError:
            print(f"Answer is not a valid {convert.__name__}")
            continue
        if answer < lower or answer > upper:
            print(f"Answer is out of bounds, must be between {lower} and {upper} inclusive")
            continue
        return answer


def get_valid_int(prompt: str, lower: float | int = float("-inf"),
                  upper: float | int = float("inf"),
                  exit_string: Optional[str] = None) -> int:
    """
    Obtains a valid integer from the user within a given range

    Args:
        prompt (str): The prompt to display to the user
        lower (float | int): The lower bound of the range
        upper (float | int): The upper bound of the range
        exit_string (str): The string to enter to exit the prompt
    Returns:
        amswer (int): The valid input entered by the user
    Raises:
        UserCancelledError: If the user cancels the input
    """
    return get_valid_num(prompt, int, lower, upper, exit_string)


def get_valid_float(prompt: str, lower: float | int = float("-inf"),
                    upper: float | int = float("inf"),
                    exit_string: Optional[str] = None) -> float:
    """
    Obtains a valid float from the user within a given range

    Args:
        prompt (str): The prompt to display to the user
        lower (float | int): The lower bound of the range
        upper (float | int): The upper bound of the range
        exit_string (str): The string to enter to exit the prompt
    Returns:
        amswer (float): The valid input entered by the user
    Raises:
        UserCancelledError: If the user cancels the input
    """
    return get_valid_num(prompt, float, lower, upper, exit_string)


def get_valid_str(prompt: str,
                  allowed_chars: Optional[list[str]] = None,
                  forbidden_chars: Optional[list[str]] = None,
                  lower: int = 0,
                  upper: int = sys.maxsize,
                  exit_string: Optional[str] = None) -> str:
    """
    Gets a valid string from the user

    Args:
        prompt (str): The prompt to display to the user
        allowed_chars (list[str]): A list of allowed characters
        forbidden_chars (list[str]): A list of forbidden characters
        lower (int): The minimum length of the string
        upper (int): The maximum length of the string
        exit_string (str): The string to enter to exit the prompt
    Returns:
        str: The valid string entered by the user
    Raises:
        UserCancelledError: If the user cancels the input
    """
    while True:
        string = input(prompt)
        if string == exit_string:
            raise UserCancelledError("User cancelled input")
        elif len(string) < lower or len(string) > upper:
            print(f"Input must be between {lower} and {upper} characters long")
            continue
        elif allowed_chars and not set(string).issubset(allowed_chars):
            print(f"Input must only contain {allowed_chars}")
            continue
        elif forbidden_chars and set(string).intersection(forbidden_chars):
            print(f"Input must not contain {forbidden_chars}")
            continue
        return string


def confirm(prompt: str = "Are you sure? [y,n]\n",
            answers: tuple[str, str] = ("y", "n")) -> bool:
    """
    Asks the user for confirmation

    Args:
        prompt (str): The prompt to display to the user
        answers (tuple[str, str]): The two valid answers

    Returns:
        bool: True if the user confirms, False otherwise

    Raises:
        ValueError: If the number of valid answers is not equal to 2
    """
    if len(answers) != 2:
        raise ValueError("There must be exactly two answers")
    while True:
        conf = input(prompt).lower()
        if conf == answers[0]:
            return True
        elif conf == answers[1]:
            return False
        print("That is not a valid answer")
