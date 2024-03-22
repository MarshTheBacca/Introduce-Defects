from typing import Optional
import sys


def get_valid_int(prompt: str, lower: float | int = float("-inf"),
                  upper: float | int = float("inf"),
                  exit_string: Optional[str] = None) -> int | None:
    """
    Obtains a valid integer from the user within a given range

    Args:
        prompt (str): The prompt to display to the user
        lower (float | int): The lower bound of the range
        upper (float | int): The upper bound of the range
        exit_string (str): The string to enter to exit the prompt

    Returns:
        int | None: The valid integer entered by the user, or None if the user entered the exit string
    """
    while True:
        answer = input(prompt)
        if answer == exit_string:
            return None
        try:
            answer = int(answer)
        except ValueError:
            print("Answer is not a valid integer")
            continue
        if answer < lower or answer > upper:
            print(f"Answer is out of bounds, must be between {lower} and {upper} inclusive")
            continue
        return answer


def get_valid_str(prompt: str,
                  allowed_chars: Optional[list[str]] = None,
                  forbidden_chars: Optional[list[str]] = None,
                  lower: int = 0,
                  upper: int = sys.maxsize,
                  verbose: bool = True) -> str:
    """
    Gets a valid string from the user

    Args:
        prompt (str): The prompt to display to the user
        allowed_chars (list[str]): A list of allowed characters
        forbidden_chars (list[str]): A list of forbidden characters
        lower (int): The minimum length of the string
        upper (int): The maximum length of the string
        verbose (bool): Whether to print error messages

    Returns:
        str: The valid string entered by the user
    """
    while True:
        string = input(prompt)
        if len(string) < lower or len(string) > upper:
            if verbose:
                print(f"Input must be between {lower} and {upper} characters long")
            continue
        if allowed_chars and not set(string).issubset(allowed_chars):
            if verbose:
                print(f"Input must only contain {allowed_chars}")
            continue
        if forbidden_chars and set(string).intersection(forbidden_chars):
            if verbose:
                print(f"Input must not contain {forbidden_chars}")
            continue
        return string
